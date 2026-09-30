// ============================================================================
// Bayesian Causal Forest Fusion (BCFusion)
//
// Research implementation of a four-component Bayesian causal forest model
// for combining an OC (primary causal) source with a supplementary UC source.
// Unit-specific likelihood powers w_i in [0, 1] control the contribution of
// each observation while preserving one exponent for the observation's entire
// likelihood contribution.
//
// Model
// -----
//   y_i = mu(x_i)
//       + D_oc_i             * mu_oc(x_i)
//       + Z_i                * tau(x_i)
//       + Z_i * D_oc_i       * tau_oc(x_i)
//       + epsilon_i,
//
// where D_oc_i = 1 denotes the OC source and Z_i is treatment. The OC-specific
// conditional treatment effect is tau(x) + tau_oc(x); the UC-specific effect is
// tau(x).
//
// Power-likelihood conventions
// ----------------------------
// * Unnormalised: f(y_i | theta)^w_i.
// * Renormalised: f(y_i | theta)^w_i / integral f(u | theta)^w_i du, w_i > 0.
//
// Under a Gaussian sampling model, the renormalised term is
// N(mu_i, sigma^2 / w_i). Both conventions use weighted residual and forest
// updates. They differ in the error-precision update: the unnormalised form
// contributes sum_i w_i likelihood degrees of freedom, while the renormalised
// form contributes one for each positive-weight observation. Rows with w_i = 0
// are excluded from fitting under either convention.
//
// Implementation details and algorithmic notes are documented in
// docs/IMPLEMENTATION_NOTES.md.
//
// [[Rcpp::plugins(cpp17)]]
// ============================================================================
#include <Rcpp.h>

#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

using namespace Rcpp;

namespace bcfusion {

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

// Uniform draw on {0, 1, ..., m-1}.
inline int rand_int(int m) {
  int k = static_cast<int>(R::runif(0.0, 1.0) * static_cast<double>(m));
  if (k >= m) k = m - 1;
  if (k < 0) k = 0;
  return k;
}

// Sorted, de-duplicated candidate split values from positive-weight rows.
// Zero-weight prediction or validation rows are not allowed to influence the
// fitted split grid. Optionally thin to at most max_cuts values.
inline std::vector< std::vector<double> >
make_cutpoints(const std::vector<double>& X, const std::vector<double>& w,
               int n, int p, int max_cuts) {

  std::vector< std::vector<double> > cuts(p);
  std::vector<double> col;

  for (int v = 0; v < p; v++) {
    col.clear();
    col.reserve(n);
    for (int i = 0; i < n; i++)
      if (w[i] > 0.0)
        col.push_back(X[static_cast<std::size_t>(v) * n + i]);
    std::sort(col.begin(), col.end());
    col.erase(std::unique(col.begin(), col.end()), col.end());

    if (max_cuts > 0 && static_cast<int>(col.size()) > max_cuts) {
      std::vector<double> thinned;
      thinned.reserve(max_cuts);
      for (int k = 0; k < max_cuts; k++) {
        std::size_t idx = static_cast<std::size_t>(
          static_cast<double>(k + 1) * col.size() / (max_cuts + 1.0));
        if (idx >= col.size()) idx = col.size() - 1;
        thinned.push_back(col[idx]);
      }
      thinned.erase(std::unique(thinned.begin(), thinned.end()), thinned.end());
      col.swap(thinned);
    }
    cuts[v].swap(col);
  }
  return cuts;
}

// ---------------------------------------------------------------------------
// Node
//
// Plain aggregate, value semantics, no R objects. Membership is NOT stored
// here -- the tree holds one integer per observation instead.
// ---------------------------------------------------------------------------

struct Node {
  double mu          = 0.0;
  double split_val   = 0.0;
  int    variable    = -1;
  int    parent      = -1;
  int    left        = -1;
  int    right       = -1;
  int    depth       = 0;
  bool   is_terminal = true;
  bool   in_use      = false;
};

// ---------------------------------------------------------------------------
// Tree
//
// Nodes live in a std::vector with explicit child indices and a free list, so
// memory is O(#nodes in use) rather than O(2^depth) as in a heap layout.
// ---------------------------------------------------------------------------

class Tree {
public:
  std::vector<Node> nodes;
  std::vector<int>  free_slots;
  std::vector<int>  node_of_obs;   // terminal node index for each observation

  // Scratch sufficient statistics, indexed by node.
  std::vector<double> sw;    // sum_i w_i
  std::vector<double> swz;   // sum_i w_i z_i
  std::vector<double> swzr;  // sum_i w_i z_i r_i

  // Scratch for subtree traversal.
  std::vector<char> mark;
  std::vector<int>  stack;

  void init(int n) {
    nodes.assign(1, Node());
    nodes[0].in_use      = true;
    nodes[0].is_terminal = true;
    nodes[0].depth       = 0;
    free_slots.clear();
    node_of_obs.assign(n, 0);
  }

  int alloc(int parent_idx, int node_depth) {
    int k;
    if (!free_slots.empty()) {
      k = free_slots.back();
      free_slots.pop_back();
    } else {
      nodes.push_back(Node());
      k = static_cast<int>(nodes.size()) - 1;
    }
    nodes[k]             = Node();
    nodes[k].parent      = parent_idx;
    nodes[k].depth       = node_depth;
    nodes[k].in_use      = true;
    nodes[k].is_terminal = true;
    return k;
  }

  int n_terminal() const {
    int c = 0;
    for (std::size_t k = 0; k < nodes.size(); k++)
      if (nodes[k].in_use && nodes[k].is_terminal) c++;
    return c;
  }

  int n_internal() const {
    int c = 0;
    for (std::size_t k = 0; k < nodes.size(); k++)
      if (nodes[k].in_use && !nodes[k].is_terminal) c++;
    return c;
  }

  // Internal nodes whose two children are both terminal.
  int n_prunable() const {
    int c = 0;
    for (std::size_t k = 0; k < nodes.size(); k++) {
      const Node& nd = nodes[k];
      if (nd.in_use && !nd.is_terminal &&
          nodes[nd.left].is_terminal && nodes[nd.right].is_terminal) c++;
    }
    return c;
  }

  int random_terminal() const {
    const int b = n_terminal();
    int pick = rand_int(b);
    for (std::size_t k = 0; k < nodes.size(); k++)
      if (nodes[k].in_use && nodes[k].is_terminal && pick-- == 0)
        return static_cast<int>(k);
    return -1;
  }

  int random_internal() const {
    const int c = n_internal();
    if (c == 0) return -1;
    int pick = rand_int(c);
    for (std::size_t k = 0; k < nodes.size(); k++)
      if (nodes[k].in_use && !nodes[k].is_terminal && pick-- == 0)
        return static_cast<int>(k);
    return -1;
  }

  int random_prunable() const {
    const int c = n_prunable();
    if (c == 0) return -1;
    int pick = rand_int(c);
    for (std::size_t k = 0; k < nodes.size(); k++) {
      const Node& nd = nodes[k];
      if (nd.in_use && !nd.is_terminal &&
          nodes[nd.left].is_terminal && nodes[nd.right].is_terminal &&
          pick-- == 0)
        return static_cast<int>(k);
    }
    return -1;
  }

  // An internal node that has a parent (so a rule can be swapped upwards).
  int random_swappable() const {
    int c = 0;
    for (std::size_t k = 0; k < nodes.size(); k++)
      if (nodes[k].in_use && !nodes[k].is_terminal && nodes[k].parent >= 0) c++;
    if (c == 0) return -1;
    int pick = rand_int(c);
    for (std::size_t k = 0; k < nodes.size(); k++)
      if (nodes[k].in_use && !nodes[k].is_terminal && nodes[k].parent >= 0 &&
          pick-- == 0)
        return static_cast<int>(k);
    return -1;
  }

  // Walk observation i down from node k until a terminal node is reached.
  int descend(int k, const double* X, int n, int i) const {
    while (!nodes[k].is_terminal) {
      const double x = X[static_cast<std::size_t>(nodes[k].variable) * n + i];
      k = (x <= nodes[k].split_val) ? nodes[k].left : nodes[k].right;
    }
    return k;
  }

  // Collect the observations currently sitting in the subtree rooted at eta,
  // together with their current terminal node (needed to revert).
  void collect_subtree(int eta,
                       std::vector<int>& obs,
                       std::vector<int>& old_assignment,
                       int n) {
    mark.assign(nodes.size(), 0);
    stack.clear();
    stack.push_back(eta);
    while (!stack.empty()) {
      const int k = stack.back();
      stack.pop_back();
      mark[k] = 1;
      if (!nodes[k].is_terminal) {
        stack.push_back(nodes[k].left);
        stack.push_back(nodes[k].right);
      }
    }
    obs.clear();
    old_assignment.clear();
    for (int i = 0; i < n; i++) {
      const int k = node_of_obs[i];
      if (mark[k]) {
        obs.push_back(i);
        old_assignment.push_back(k);
      }
    }
  }

  // One O(n) pass. No allocation beyond the (reused) scratch resize.
  void accumulate(const double* w, const double* z, const double* r, int n) {
    sw.assign(nodes.size(), 0.0);
    swz.assign(nodes.size(), 0.0);
    swzr.assign(nodes.size(), 0.0);
    for (int i = 0; i < n; i++) {
      const int    k   = node_of_obs[i];
      const double wi  = w[i];
      const double wzi = wi * z[i];
      sw[k]   += wi;
      swz[k]  += wzi;
      swzr[k] += wzi * r[i];
    }
  }

  // Marginal log-likelihood plus tree-structure prior, with the two terms that
  // cancel in the MH ratio -- (nj/2)log(tau) and -(tau/2) sum w r^2 -- omitted.
  // Requires accumulate() to have been called for the current membership.
  double log_lik(double tau_node, double tau_prec,
                 double alpha, double beta) const {
    double ll = 0.0;
    for (std::size_t k = 0; k < nodes.size(); k++) {
      const Node& nd = nodes[k];
      if (!nd.in_use) continue;
      const double d = static_cast<double>(nd.depth);
      if (nd.is_terminal) {
        const double prec = tau_node + swz[k] * tau_prec;
        ll += 0.5 * std::log(tau_node / prec)
            + 0.5 * tau_prec * tau_prec * swzr[k] * swzr[k] / prec
            + std::log1p(-alpha * std::pow(1.0 + d, -beta));
      } else {
        ll += std::log(alpha) - beta * std::log(1.0 + d);
      }
    }
    return ll;
  }

  // Weighted node-size constraint. min_w guards the effective sample size of
  // the node; min_wz guards the effective number of ACTIVE units (treated, or
  // OC source, depending on the forest's indicator).
  bool nodes_too_small(double min_w, double min_wz) const {
    for (std::size_t k = 0; k < nodes.size(); k++) {
      const Node& nd = nodes[k];
      if (nd.in_use && nd.is_terminal) {
        if (sw[k] < min_w || swz[k] < min_wz) return true;
      }
    }
    return false;
  }

  void draw_mu(double tau_node, double tau_prec) {
    for (std::size_t k = 0; k < nodes.size(); k++) {
      Node& nd = nodes[k];
      if (nd.in_use && nd.is_terminal) {
        const double prec = tau_node + swz[k] * tau_prec;
        nd.mu = R::rnorm(tau_prec * swzr[k] / prec, std::sqrt(1.0 / prec));
      }
    }
  }

  void predict_into(double* out, int n) const {
    for (int i = 0; i < n; i++) out[i] = nodes[node_of_obs[i]].mu;
  }
};

// ---------------------------------------------------------------------------
// Proposals
//
// Every move is applied in place and reverted if rejected, so no tree is ever
// copied. The Proposal object is reused across trees to keep its vectors'
// capacity and avoid per-proposal allocation.
// ---------------------------------------------------------------------------

enum MoveType { MOVE_GROW = 0, MOVE_PRUNE = 1, MOVE_CHANGE = 2, MOVE_SWAP = 3 };

struct Proposal {
  MoveType type      = MOVE_GROW;
  double   log_q     = 0.0;   // log q(T | T*) - log q(T* | T)
  int      node      = -1;
  int      left      = -1;
  int      right     = -1;
  int      other     = -1;
  int      old_var   = -1;
  double   old_split = 0.0;

  std::vector<int> touched;
  std::vector<int> old_assignment;

  void reset() {
    log_q = 0.0;
    node = left = right = other = -1;
    old_var = -1;
    old_split = 0.0;
    touched.clear();
    old_assignment.clear();
  }
};

// Pick a variable and a split value valid for the observation set `obs`.
// Candidates are the precomputed cutpoints lying in [min, max) of the node's
// values for that variable, which guarantees both children are non-empty.
inline bool draw_rule(const std::vector<int>& obs,
                      const double* X, const double* w, int n, int p,
                      const std::vector< std::vector<double> >& cuts,
                      int& v_out, double& c_out) {

  if (obs.size() < 2) return false;

  const int v = rand_int(p);
  double lo =  std::numeric_limits<double>::infinity();
  double hi = -std::numeric_limits<double>::infinity();
  const double* xcol = X + static_cast<std::size_t>(v) * n;
  for (std::size_t t = 0; t < obs.size(); t++) {
    const int i = obs[t];
    if (w[i] <= 0.0) continue;
    const double x = xcol[i];
    if (x < lo) lo = x;
    if (x > hi) hi = x;
  }
  if (!(lo < hi)) return false;

  const std::vector<double>& cv = cuts[v];
  const std::vector<double>::const_iterator it_lo =
    std::lower_bound(cv.begin(), cv.end(), lo);
  const std::vector<double>::const_iterator it_hi =
    std::lower_bound(cv.begin(), cv.end(), hi);
  const int ncand = static_cast<int>(it_hi - it_lo);
  if (ncand <= 0) return false;

  v_out = v;
  c_out = *(it_lo + rand_int(ncand));
  return true;
}

// Return true if terminal node `leaf` is contained in the subtree rooted at
// `ancestor`.
inline bool terminal_descends_from(const Tree& T, int leaf, int ancestor) {
  int k = leaf;
  while (k >= 0) {
    if (k == ancestor) return true;
    k = T.nodes[k].parent;
  }
  return false;
}

// Number of positive-weight split candidates available at one node. This is
// needed for the split-rule prior ratio of a SWAP proposal.
inline int rule_support_at_node(
    const Tree& T, int node, int variable, double split_value,
    bool check_split, const double* X, const double* w, int n,
    const std::vector< std::vector<double> >& cuts,
    bool& split_is_valid) {

  double lo =  std::numeric_limits<double>::infinity();
  double hi = -std::numeric_limits<double>::infinity();
  int n_obs = 0;
  const double* xcol = X + static_cast<std::size_t>(variable) * n;
  for (int i = 0; i < n; i++) {
    if (w[i] > 0.0 && terminal_descends_from(T, T.node_of_obs[i], node)) {
      const double x = xcol[i];
      if (x < lo) lo = x;
      if (x > hi) hi = x;
      n_obs++;
    }
  }

  split_is_valid = false;
  if (n_obs < 2 || !(lo < hi)) return 0;
  const std::vector<double>& cv = cuts[variable];
  const auto it_lo = std::lower_bound(cv.begin(), cv.end(), lo);
  const auto it_hi = std::lower_bound(cv.begin(), cv.end(), hi);
  const int n_cand = static_cast<int>(it_hi - it_lo);
  if (n_cand <= 0) return 0;
  if (!check_split) {
    split_is_valid = true;
    return n_cand;
  }
  const auto it = std::lower_bound(it_lo, it_hi, split_value);
  split_is_valid = (it != it_hi && *it == split_value);
  return n_cand;
}

inline double log_rule_prior_subtree(
    const Tree& T, int root, const double* X, const double* w, int n,
    const std::vector< std::vector<double> >& cuts) {

  double out = 0.0;
  for (std::size_t kk = 0; kk < T.nodes.size(); kk++) {
    const int k = static_cast<int>(kk);
    const Node& nd = T.nodes[k];
    if (!nd.in_use || nd.is_terminal) continue;
    int a = k;
    bool in_subtree = false;
    while (a >= 0) {
      if (a == root) { in_subtree = true; break; }
      a = T.nodes[a].parent;
    }
    if (!in_subtree) continue;
    bool valid = false;
    const int ncand = rule_support_at_node(
      T, k, nd.variable, nd.split_val, true, X, w, n, cuts, valid);
    if (!valid || ncand <= 0)
      return -std::numeric_limits<double>::infinity();
    out -= std::log(static_cast<double>(ncand));
  }
  return out;
}

inline bool propose_grow(Tree& T, const double* X, const double* w,
                         int n, int p,
                         const std::vector< std::vector<double> >& cuts,
                         Proposal& pr) {

  const int eta = T.random_terminal();
  if (eta < 0) return false;

  const int b = T.n_terminal();

  pr.touched.clear();
  for (int i = 0; i < n; i++)
    if (T.node_of_obs[i] == eta) pr.touched.push_back(i);

  int v; double c;
  if (!draw_rule(pr.touched, X, w, n, p, cuts, v, c)) return false;

  const int d = T.nodes[eta].depth;
  const int l = T.alloc(eta, d + 1);
  const int r = T.alloc(eta, d + 1);

  T.nodes[eta].variable    = v;
  T.nodes[eta].split_val   = c;
  T.nodes[eta].is_terminal = false;
  T.nodes[eta].left        = l;
  T.nodes[eta].right       = r;

  const double* xcol = X + static_cast<std::size_t>(v) * n;
  for (std::size_t t = 0; t < pr.touched.size(); t++) {
    const int i = pr.touched[t];
    T.node_of_obs[i] = (xcol[i] <= c) ? l : r;
  }

  pr.type  = MOVE_GROW;
  pr.node  = eta;
  pr.left  = l;
  pr.right = r;
  // Proposal ratio: the 1/p and 1/n_cut factors cancel against the splitting
  // rule prior, leaving only the node-count terms.
  pr.log_q = std::log(static_cast<double>(b))
           - std::log(static_cast<double>(T.n_prunable()));
  return true;
}

inline void revert_grow(Tree& T, const Proposal& pr) {
  for (std::size_t t = 0; t < pr.touched.size(); t++)
    T.node_of_obs[pr.touched[t]] = pr.node;
  T.nodes[pr.left].in_use  = false;
  T.nodes[pr.right].in_use = false;
  T.free_slots.push_back(pr.left);
  T.free_slots.push_back(pr.right);
  T.nodes[pr.node].is_terminal = true;
  T.nodes[pr.node].left        = -1;
  T.nodes[pr.node].right       = -1;
  T.nodes[pr.node].variable    = -1;
}

inline bool propose_prune(Tree& T, int n, Proposal& pr) {

  const int eta = T.random_prunable();
  if (eta < 0) return false;

  const int w = T.n_prunable();
  const int l = T.nodes[eta].left;
  const int r = T.nodes[eta].right;

  pr.touched.clear();
  for (int i = 0; i < n; i++) {
    const int k = T.node_of_obs[i];
    if (k == l || k == r) {
      pr.touched.push_back(i);
      T.node_of_obs[i] = eta;
    }
  }

  T.nodes[l].in_use        = false;
  T.nodes[r].in_use        = false;
  T.nodes[eta].is_terminal = true;
  // eta's variable / split_val are deliberately left intact so a rejection can
  // re-partition exactly.

  pr.type  = MOVE_PRUNE;
  pr.node  = eta;
  pr.left  = l;
  pr.right = r;
  pr.log_q = std::log(static_cast<double>(w))
           - std::log(static_cast<double>(T.n_terminal()));
  return true;
}

inline void accept_prune(Tree& T, const Proposal& pr) {
  T.free_slots.push_back(pr.left);
  T.free_slots.push_back(pr.right);
  T.nodes[pr.node].left     = -1;
  T.nodes[pr.node].right    = -1;
  T.nodes[pr.node].variable = -1;
}

inline void revert_prune(Tree& T, const Proposal& pr, const double* X, int n) {
  Node& e = T.nodes[pr.node];
  e.is_terminal            = false;
  T.nodes[pr.left].in_use  = true;
  T.nodes[pr.right].in_use = true;
  const double* xcol = X + static_cast<std::size_t>(e.variable) * n;
  const double c = e.split_val;
  for (std::size_t t = 0; t < pr.touched.size(); t++) {
    const int i = pr.touched[t];
    T.node_of_obs[i] = (xcol[i] <= c) ? pr.left : pr.right;
  }
}

inline bool propose_change(Tree& T, const double* X, const double* w,
                           int n, int p,
                           const std::vector< std::vector<double> >& cuts,
                           Proposal& pr) {

  const int eta = T.random_internal();
  if (eta < 0) return false;

  T.collect_subtree(eta, pr.touched, pr.old_assignment, n);

  int v; double c;
  if (!draw_rule(pr.touched, X, w, n, p, cuts, v, c)) return false;

  pr.old_var   = T.nodes[eta].variable;
  pr.old_split = T.nodes[eta].split_val;
  T.nodes[eta].variable  = v;
  T.nodes[eta].split_val = c;

  for (std::size_t t = 0; t < pr.touched.size(); t++) {
    const int i = pr.touched[t];
    T.node_of_obs[i] = T.descend(eta, X, n, i);
  }

  pr.type  = MOVE_CHANGE;
  pr.node  = eta;
  // Structure is unchanged, so the tree prior cancels; the rule prior cancels
  // against the proposal under the "uniform over available cutpoints in the
  // node" convention. Symmetric.
  pr.log_q = 0.0;
  return true;
}

inline void revert_change(Tree& T, const Proposal& pr) {
  T.nodes[pr.node].variable  = pr.old_var;
  T.nodes[pr.node].split_val = pr.old_split;
  for (std::size_t t = 0; t < pr.touched.size(); t++)
    T.node_of_obs[pr.touched[t]] = pr.old_assignment[t];
}

inline bool propose_swap(Tree& T, const double* X, const double* w, int n,
                         const std::vector< std::vector<double> >& cuts,
                         Proposal& pr) {

  const int child = T.random_swappable();
  if (child < 0) return false;
  const int par = T.nodes[child].parent;

  const double old_rule_prior =
    log_rule_prior_subtree(T, par, X, w, n, cuts);
  if (!std::isfinite(old_rule_prior)) return false;

  T.collect_subtree(par, pr.touched, pr.old_assignment, n);

  std::swap(T.nodes[par].variable,  T.nodes[child].variable);
  std::swap(T.nodes[par].split_val, T.nodes[child].split_val);

  for (std::size_t t = 0; t < pr.touched.size(); t++) {
    const int i = pr.touched[t];
    T.node_of_obs[i] = T.descend(par, X, n, i);
  }

  const double new_rule_prior =
    log_rule_prior_subtree(T, par, X, w, n, cuts);

  pr.type  = MOVE_SWAP;
  pr.node  = par;
  pr.other = child;
  pr.log_q = std::isfinite(new_rule_prior)
    ? new_rule_prior - old_rule_prior
    : -std::numeric_limits<double>::infinity();
  return true;
}

inline void revert_swap(Tree& T, const Proposal& pr) {
  std::swap(T.nodes[pr.node].variable,  T.nodes[pr.other].variable);
  std::swap(T.nodes[pr.node].split_val, T.nodes[pr.other].split_val);
  for (std::size_t t = 0; t < pr.touched.size(); t++)
    T.node_of_obs[pr.touched[t]] = pr.old_assignment[t];
}

// ---------------------------------------------------------------------------
// One Gibbs sweep over a forest
//
// A single routine drives all four forests. The mu forest passes an all-ones
// indicator, which reduces it to the standard BART update -- so the two
// near-duplicate code paths of the original collapse into one.
// ---------------------------------------------------------------------------

struct ForestCfg {
  int    n_tree   = 0;
  double alpha    = 0.95;
  double beta     = 2.0;
  double tau_node = 1.0;      // terminal node prior precision
  double min_w    = 5.0;      // minimum weighted node size
  double min_wz   = 0.0;      // minimum weighted count of active units
  const double* X = nullptr;  // n x p, column-major
  int    p        = 0;
  const std::vector< std::vector<double> >* cuts = nullptr;
  const double* z = nullptr;  // indicator
  const double* w = nullptr;  // power-posterior weights
};

inline void update_forest(std::vector<Tree>& trees,
                          std::vector<double>& tree_preds,  // n * n_tree
                          std::vector<double>& fit,         // n
                          std::vector<double>& resid,       // n
                          std::vector<double>& newcol,      // n, scratch
                          Proposal& pr,                     // scratch
                          const ForestCfg& cfg,
                          double tau_prec, int n,
                          long long* prop, long long* acc) {

  const double* z = cfg.z;
  const double* w = cfg.w;

  for (int j = 0; j < cfg.n_tree; j++) {

    double* col = &tree_preds[static_cast<std::size_t>(j) * n];

    // Remove this tree's contribution from the running fit and residual.
    for (int i = 0; i < n; i++) {
      const double d = col[i];
      fit[i]   -= d;
      resid[i] += z[i] * d;
    }

    Tree& T = trees[j];

    T.accumulate(w, z, resid.data(), n);
    const double lold = T.log_lik(cfg.tau_node, tau_prec, cfg.alpha, cfg.beta);

    pr.reset();
    const int move = rand_int(4);
    bool applied = false;

    switch (move) {
      case MOVE_GROW:
        applied = propose_grow(T, cfg.X, w, n, cfg.p, *cfg.cuts, pr);
        break;
      case MOVE_PRUNE:
        applied = propose_prune(T, n, pr);
        break;
      case MOVE_CHANGE:
        applied = propose_change(T, cfg.X, w, n, cfg.p, *cfg.cuts, pr);
        break;
      default:
        applied = propose_swap(T, cfg.X, w, n, *cfg.cuts, pr);
        break;
    }

    if (applied) {
      prop[move]++;

      T.accumulate(w, z, resid.data(), n);

      bool reject = T.nodes_too_small(cfg.min_w, cfg.min_wz);
      if (!reject) {
        const double lnew =
          T.log_lik(cfg.tau_node, tau_prec, cfg.alpha, cfg.beta);
        reject = !(lnew - lold + pr.log_q > std::log(R::runif(0.0, 1.0)));
      }

      if (reject) {
        switch (pr.type) {
          case MOVE_GROW:   revert_grow(T, pr);              break;
          case MOVE_PRUNE:  revert_prune(T, pr, cfg.X, n);   break;
          case MOVE_CHANGE: revert_change(T, pr);            break;
          default:          revert_swap(T, pr);              break;
        }
        T.accumulate(w, z, resid.data(), n);  // restore stats for the mu draw
      } else {
        acc[move]++;
        if (pr.type == MOVE_PRUNE) accept_prune(T, pr);
      }
    }

    T.draw_mu(cfg.tau_node, tau_prec);
    T.predict_into(newcol.data(), n);

    for (int i = 0; i < n; i++) {
      const double d = newcol[i];
      col[i]    = d;
      fit[i]   += d;
      resid[i] -= z[i] * d;
    }
  }
}

// ---------------------------------------------------------------------------
// Driver
//
// Deliberately free of R types so it can be compiled and tested standalone.
// ---------------------------------------------------------------------------

struct BcfInput {
  int n = 0, p = 0, p_tau = 0;
  std::vector<double> X, X_tau, y;
  std::vector<double> d_oc, z;
  std::vector<double> w;
  double alpha_mu = 0.95, beta_mu = 2.0;
  double alpha_mu_oc = 0.65, beta_mu_oc = 3.0;
  double alpha_tau = 0.35, beta_tau = 3.0;
  double alpha_tau_oc = 0.15, beta_tau_oc = 3.0;
  double tau_mu = 1.0, tau_mu_oc = 1.0, tau_tau = 1.0, tau_tau_oc = 1.0;
  double nu = 3.0, lambda = 0.1;
  int n_iter = 1000, n_burn = 0, thin = 1;
  int n_tree_mu = 100, n_tree_mu_oc = 50, n_tree_tau = 50, n_tree_tau_oc = 25;
  double min_node_weight = 5.0, min_active_weight = 5.0;
  int max_cuts = 0;
  bool renormalise_likelihood = true;
  bool store_draws = true, return_trees = false;
};

struct BcfOutput {
  int n_kept = 0;
  std::vector<double> preds_mu, preds_mu_oc, preds_tau, preds_tau_oc;
  std::vector<double> error_precisions, error_sds;
  std::vector<double> mu_hat, mu_oc_hat, tau_hat, tau_oc_hat;
  std::vector<long long> prop, acc;   // 4 forests x 4 moves, row-major
  // Optional final MCMC-state forests. These are returned only for inspection;
  // posterior summaries continue to use all retained prediction draws.
  std::vector<Tree> final_mu, final_mu_oc, final_tau, final_tau_oc;
  double y_mean = 0.0, y_sd = 1.0;
};

inline BcfOutput run_bcf(const BcfInput& in, bool (*interrupt_check)() = nullptr) {

  const int n = in.n;

  BcfOutput out;
  out.prop.assign(16, 0);
  out.acc.assign(16, 0);

  // -- scale y ---------------------------------------------------------------
  // Anchor the transformation to positive-weight OC rows. Thus eta changes
  // only the UC likelihood contribution, not the outcome transformation and
  // induced tree-prior scale. A deliberately UC-only comparator falls back to
  // its positive-weight rows. Zero-weight rows are never used.
  int n_positive_oc = 0;
  for (int i = 0; i < n; i++)
    if (in.d_oc[i] == 1.0 && in.w[i] > 0.0) n_positive_oc++;
  const bool use_oc_scale = n_positive_oc >= 2;

  int n_scale = 0;
  double ymean = 0.0;
  for (int i = 0; i < n; i++) {
    const bool scale_row = use_oc_scale
      ? (in.d_oc[i] == 1.0 && in.w[i] > 0.0)
      : (in.w[i] > 0.0);
    if (scale_row) {
      ymean += in.y[i];
      n_scale++;
    }
  }
  if (n_scale < 2)
    throw std::runtime_error(
      "At least two positive-weight observations are required for scaling.");
  ymean /= static_cast<double>(n_scale);
  double ss = 0.0;
  for (int i = 0; i < n; i++) {
    const bool scale_row = use_oc_scale
      ? (in.d_oc[i] == 1.0 && in.w[i] > 0.0)
      : (in.w[i] > 0.0);
    if (scale_row) {
      const double d = in.y[i] - ymean;
      ss += d * d;
    }
  }
  const double ysd = std::sqrt(ss / static_cast<double>(n_scale - 1));
  if (!(ysd > 0.0) || !std::isfinite(ysd))
    throw std::runtime_error("OC outcome scale is not finite and positive.");
  out.y_mean = ymean;
  out.y_sd   = ysd;

  std::vector<double> ys(n);
  for (int i = 0; i < n; i++) ys[i] = (in.y[i] - ymean) / ysd;

  // -- indicators ------------------------------------------------------------
  std::vector<double> ones(n, 1.0);
  std::vector<double> z_oc(n);
  for (int i = 0; i < n; i++) z_oc[i] = in.z[i] * in.d_oc[i];

  // -- cutpoints (built once) ------------------------------------------------
  const std::vector< std::vector<double> > cuts =
    make_cutpoints(in.X, in.w, n, in.p, in.max_cuts);
  const std::vector< std::vector<double> > cuts_tau =
    make_cutpoints(in.X_tau, in.w, n, in.p_tau, in.max_cuts);

  // -- forests ---------------------------------------------------------------
  std::vector<Tree> f_mu(in.n_tree_mu);
  std::vector<Tree> f_mu_oc(in.n_tree_mu_oc);
  std::vector<Tree> f_tau(in.n_tree_tau);
  std::vector<Tree> f_tau_oc(in.n_tree_tau_oc);
  for (int j = 0; j < in.n_tree_mu;      j++) f_mu[j].init(n);
  for (int j = 0; j < in.n_tree_mu_oc;  j++) f_mu_oc[j].init(n);
  for (int j = 0; j < in.n_tree_tau;     j++) f_tau[j].init(n);
  for (int j = 0; j < in.n_tree_tau_oc; j++) f_tau_oc[j].init(n);

  std::vector<double> tp_mu     (static_cast<std::size_t>(n) * in.n_tree_mu,      0.0);
  std::vector<double> tp_mu_oc (static_cast<std::size_t>(n) * in.n_tree_mu_oc,  0.0);
  std::vector<double> tp_tau    (static_cast<std::size_t>(n) * in.n_tree_tau,     0.0);
  std::vector<double> tp_tau_oc(static_cast<std::size_t>(n) * in.n_tree_tau_oc, 0.0);

  std::vector<double> fit_mu(n, 0.0), fit_mu_oc(n, 0.0);
  std::vector<double> fit_tau(n, 0.0), fit_tau_oc(n, 0.0);
  std::vector<double> resid(ys);
  std::vector<double> newcol(n, 0.0);
  Proposal pr;

  double tau_prec = 1.0;

  ForestCfg c_mu;
  c_mu.n_tree = in.n_tree_mu; c_mu.alpha = in.alpha_mu; c_mu.beta = in.beta_mu;
  c_mu.tau_node = in.tau_mu;  c_mu.min_w = in.min_node_weight; c_mu.min_wz = 0.0;
  c_mu.X = in.X.data(); c_mu.p = in.p; c_mu.cuts = &cuts;
  c_mu.z = ones.data(); c_mu.w = in.w.data();

  ForestCfg c_mu_oc;
  c_mu_oc.n_tree = in.n_tree_mu_oc; c_mu_oc.alpha = in.alpha_mu_oc;
  c_mu_oc.beta = in.beta_mu_oc;     c_mu_oc.tau_node = in.tau_mu_oc;
  c_mu_oc.min_w = in.min_node_weight;   c_mu_oc.min_wz = in.min_active_weight;
  c_mu_oc.X = in.X.data(); c_mu_oc.p = in.p; c_mu_oc.cuts = &cuts;
  c_mu_oc.z = in.d_oc.data(); c_mu_oc.w = in.w.data();

  ForestCfg c_tau;
  c_tau.n_tree = in.n_tree_tau; c_tau.alpha = in.alpha_tau; c_tau.beta = in.beta_tau;
  c_tau.tau_node = in.tau_tau;  c_tau.min_w = in.min_node_weight; c_tau.min_wz = in.min_active_weight;
  c_tau.X = in.X_tau.data(); c_tau.p = in.p_tau; c_tau.cuts = &cuts_tau;
  c_tau.z = in.z.data(); c_tau.w = in.w.data();

  ForestCfg c_tau_oc;
  c_tau_oc.n_tree = in.n_tree_tau_oc; c_tau_oc.alpha = in.alpha_tau_oc;
  c_tau_oc.beta = in.beta_tau_oc;     c_tau_oc.tau_node = in.tau_tau_oc;
  c_tau_oc.min_w = in.min_node_weight;    c_tau_oc.min_wz = in.min_active_weight;
  c_tau_oc.X = in.X_tau.data(); c_tau_oc.p = in.p_tau; c_tau_oc.cuts = &cuts_tau;
  c_tau_oc.z = z_oc.data(); c_tau_oc.w = in.w.data();

  // -- output sizing ---------------------------------------------------------
  const int n_post = (in.n_iter > in.n_burn) ? (in.n_iter - in.n_burn) : 0;
  const int n_kept = (n_post + in.thin - 1) / in.thin;
  out.n_kept = n_kept;
  out.error_precisions.reserve(n_kept);
  out.error_sds.reserve(n_kept);
  if (in.store_draws) {
    out.preds_mu.assign     (static_cast<std::size_t>(n) * n_kept, 0.0);
    out.preds_mu_oc.assign  (static_cast<std::size_t>(n) * n_kept, 0.0);
    out.preds_tau.assign    (static_cast<std::size_t>(n) * n_kept, 0.0);
    out.preds_tau_oc.assign (static_cast<std::size_t>(n) * n_kept, 0.0);
  }
  out.mu_hat.assign(n, 0.0);
  out.mu_oc_hat.assign(n, 0.0);
  out.tau_hat.assign(n, 0.0);
  out.tau_oc_hat.assign(n, 0.0);

  double sum_w_prec = 0.0;
  int n_positive_weight = 0;
  for (int i = 0; i < n; i++) {
    sum_w_prec += in.w[i];
    if (in.w[i] > 0.0) n_positive_weight++;
  }

  int kept = 0;

  // -- MCMC ------------------------------------------------------------------
  for (int iter = 0; iter < in.n_iter; iter++) {

    update_forest(f_mu,      tp_mu,      fit_mu,      resid, newcol, pr,
                  c_mu,      tau_prec, n, &out.prop[0],  &out.acc[0]);
    update_forest(f_mu_oc,  tp_mu_oc,  fit_mu_oc,  resid, newcol, pr,
                  c_mu_oc,  tau_prec, n, &out.prop[4],  &out.acc[4]);
    update_forest(f_tau,     tp_tau,     fit_tau,     resid, newcol, pr,
                  c_tau,     tau_prec, n, &out.prop[8],  &out.acc[8]);
    update_forest(f_tau_oc, tp_tau_oc, fit_tau_oc, resid, newcol, pr,
                  c_tau_oc, tau_prec, n, &out.prop[12], &out.acc[12]);

    // Recompute the residual from the component fits once per sweep so that
    // the incremental add/subtract above cannot accumulate drift.
    for (int i = 0; i < n; i++) {
      resid[i] = ys[i] - fit_mu[i]
               - in.d_oc[i]   * fit_mu_oc[i]
               - in.z[i] * fit_tau[i]
               - z_oc[i] * fit_tau_oc[i];
    }

    // Error precision. For the ordinary power posterior, sum(w_i) replaces n.
    // For the renormalised Gaussian power likelihood, each positive-weight
    // density has normalising factor sqrt(w_i * tau / (2*pi)), so the log(tau)
    // coefficient is n_positive_weight / 2. The sum(log(w_i))/2 term is fixed
    // and therefore drops from this conditional update.
    double S = 0.0;
    for (int i = 0; i < n; i++) S += in.w[i] * resid[i] * resid[i];
    const double likelihood_df = in.renormalise_likelihood
      ? static_cast<double>(n_positive_weight)
      : sum_w_prec;
    const double shape = (likelihood_df + in.nu) / 2.0;
    const double rate  = (S + in.nu * in.lambda) / 2.0;
    tau_prec = R::rgamma(shape, 1.0 / rate);

    // Store.
    if (iter >= in.n_burn && ((iter - in.n_burn) % in.thin == 0) && kept < n_kept) {
      out.error_precisions.push_back(tau_prec / (ysd * ysd));
      out.error_sds.push_back(ysd / std::sqrt(tau_prec));
      for (int i = 0; i < n; i++) {
        const double a = fit_mu[i]      * ysd + ymean;
        const double b = fit_mu_oc[i]  * ysd;
        const double c = fit_tau[i]     * ysd;
        const double d = fit_tau_oc[i] * ysd;
        out.mu_hat[i]      += a;
        out.mu_oc_hat[i]  += b;
        out.tau_hat[i]     += c;
        out.tau_oc_hat[i] += d;
        if (in.store_draws) {
          const std::size_t off = static_cast<std::size_t>(kept) * n + i;
          out.preds_mu[off]      = a;
          out.preds_mu_oc[off]  = b;
          out.preds_tau[off]     = c;
          out.preds_tau_oc[off] = d;
        }
      }
      kept++;
    }

    if (interrupt_check && (iter % 50 == 0)) interrupt_check();
  }

  if (kept > 0) {
    const double inv = 1.0 / kept;
    for (int i = 0; i < n; i++) {
      out.mu_hat[i]      *= inv;
      out.mu_oc_hat[i]  *= inv;
      out.tau_hat[i]     *= inv;
      out.tau_oc_hat[i] *= inv;
    }
  }

  if (in.return_trees) {
    out.final_mu      = std::move(f_mu);
    out.final_mu_oc  = std::move(f_mu_oc);
    out.final_tau     = std::move(f_tau);
    out.final_tau_oc = std::move(f_tau_oc);
  }

  return out;
}

}  // namespace bcfusion

// ---------------------------------------------------------------------------
// Rcpp boundary
// ---------------------------------------------------------------------------

static bool bcf_interrupt() {
  Rcpp::checkUserInterrupt();
  return false;
}

DataFrame serialize_tree(const bcfusion::Tree& tree,
                         const CharacterVector& feature_names,
                         double outcome_scale) {
  const int n_nodes = static_cast<int>(tree.nodes.size());
  IntegerVector node_id(n_nodes), parent(n_nodes), left(n_nodes), right(n_nodes);
  IntegerVector depth(n_nodes), n_obs(n_nodes), variable_index(n_nodes);
  LogicalVector terminal(n_nodes), in_use(n_nodes);
  NumericVector split_value(n_nodes), leaf_value(n_nodes);
  CharacterVector variable(n_nodes);

  std::fill(n_obs.begin(), n_obs.end(), 0);
  for (std::size_t i = 0; i < tree.node_of_obs.size(); i++) {
    int k = tree.node_of_obs[i];
    while (k >= 0) {
      n_obs[k]++;
      k = tree.nodes[k].parent;
    }
  }

  for (int k = 0; k < n_nodes; k++) {
    const bcfusion::Node& nd = tree.nodes[k];
    node_id[k] = k + 1;
    parent[k] = nd.parent >= 0 ? nd.parent + 1 : NA_INTEGER;
    left[k] = nd.left >= 0 ? nd.left + 1 : NA_INTEGER;
    right[k] = nd.right >= 0 ? nd.right + 1 : NA_INTEGER;
    depth[k] = nd.depth;
    terminal[k] = nd.is_terminal;
    in_use[k] = nd.in_use;
    if (nd.variable >= 0) {
      variable_index[k] = nd.variable + 1;
      variable[k] = feature_names[nd.variable];
      split_value[k] = nd.split_val;
    } else {
      variable_index[k] = NA_INTEGER;
      variable[k] = NA_STRING;
      split_value[k] = NA_REAL;
    }
    leaf_value[k] = nd.is_terminal ? nd.mu * outcome_scale : NA_REAL;
  }

  return DataFrame::create(
    Named("node_id") = node_id,
    Named("parent") = parent,
    Named("left") = left,
    Named("right") = right,
    Named("depth") = depth,
    Named("terminal") = terminal,
    Named("in_use") = in_use,
    Named("variable_index") = variable_index,
    Named("variable") = variable,
    Named("split_value") = split_value,
    Named("leaf_value") = leaf_value,
    Named("n_obs") = n_obs
  );
}

List serialize_forest(const std::vector<bcfusion::Tree>& forest,
                      const CharacterVector& feature_names,
                      double outcome_scale) {
  List out(forest.size());
  for (std::size_t j = 0; j < forest.size(); j++) {
    out[j] = serialize_tree(forest[j], feature_names, outcome_scale);
  }
  return out;
}

// Fit Bayesian Causal Forest Fusion (BCFusion) by MCMC.
//
// Required inputs:
//   X             prognostic covariate matrix (n x p)
//   y             observed outcome
//   D_oc          source indicator: 1 = OC, 0 = UC
//   Z             treatment indicator: 1 = treated, 0 = control
//   X_tau         treatment-effect covariate matrix (n x p_tau)
//   power_weights unit-specific likelihood powers in [0, 1]
//
// Positive-weight OC observations must have power one. UC observations may
// have powers between zero and one. A zero-weight row is excluded from fitting
// but can still receive fitted component values from the final tree ensemble.
//
// The *_oc quantities returned by the internal sampler are source-deviation
// components. The public return object therefore labels them explicitly as
// deviations and also returns cate_uc_hat = tau_hat and
// cate_oc_hat = tau_hat + tau_oc_deviation_hat.
//
// [[Rcpp::export]]
List bcfusion_fit(NumericMatrix X,
                  NumericVector y,
                  NumericVector D_oc,
                  NumericVector Z,
                  NumericMatrix X_tau,
                  NumericVector power_weights,
                  double alpha_mu = 0.95,
                  double beta_mu = 2.0,
                  double alpha_mu_oc = 0.65,
                  double beta_mu_oc = 3.0,
                  double alpha_tau = 0.35,
                  double beta_tau = 3.0,
                  double alpha_tau_oc = 0.15,
                  double beta_tau_oc = 3.0,
                  double tau_mu = 1.0,
                  double tau_mu_oc = 1.0,
                  double tau_tau = 1.0,
                  double tau_tau_oc = 1.0,
                  double nu = 3.0,
                  double lambda = 0.1,
                  int n_iter = 1000,
                  int n_burn = 0,
                  int thin = 1,
                  int n_tree_mu = 100,
                  int n_tree_mu_oc = 50,
                  int n_tree_tau = 50,
                  int n_tree_tau_oc = 25,
                  double min_node_weight = 5.0,
                  double min_active_weight = 5.0,
                  int max_cuts = 0,
                  bool renormalise_likelihood = true,
                  bool store_draws = true,
                  bool return_trees = false,
                  bool verbose = true) {

  const int n = y.size();

  if (n < 5)               Rcpp::stop("`y` must have at least 5 observations.");
  if (X.nrow() != n)       Rcpp::stop("`X` must have nrow(X) == length(y).");
  if (X_tau.nrow() != n)   Rcpp::stop("`X_tau` must have nrow(X_tau) == length(y).");
  if (D_oc.size() != n)     Rcpp::stop("`D_oc` must have length == length(y).");
  if (Z.size() != n)        Rcpp::stop("`Z` must have length == length(y).");
  if (power_weights.size() != n) Rcpp::stop("`power_weights` has the wrong length.");
  double supplied_weight_sum = 0.0;
  for (int i = 0; i < n; i++) {
    if (!std::isfinite(power_weights[i]) || power_weights[i] < 0.0 || power_weights[i] > 1.0)
      Rcpp::stop("`power_weights` must contain finite values in [0, 1].");
    supplied_weight_sum += power_weights[i];
  }
  if (!(supplied_weight_sum > 1.0))
    Rcpp::stop("`power_weights` must have total greater than one.");
  if (n_iter < 1)          Rcpp::stop("`n_iter` must be at least 1.");
  if (thin < 1)            Rcpp::stop("`thin` must be at least 1.");
  if (n_burn < 0 || n_burn >= n_iter) Rcpp::stop("`n_burn` must be in [0, n_iter).");
  if (X.ncol() < 1 || X_tau.ncol() < 1)
    Rcpp::stop("`X` and `X_tau` must each contain at least one column.");
  if (n_tree_mu < 1 || n_tree_tau < 1)
    Rcpp::stop("The shared prognostic and treatment forests need at least one tree.");
  if (n_tree_mu_oc < 0 || n_tree_tau_oc < 0)
    Rcpp::stop("Source-specific forest tree counts must be non-negative.");
  if (!(alpha_mu > 0.0 && alpha_mu < 1.0) ||
      !(alpha_mu_oc > 0.0 && alpha_mu_oc < 1.0) ||
      !(alpha_tau > 0.0 && alpha_tau < 1.0) ||
      !(alpha_tau_oc > 0.0 && alpha_tau_oc < 1.0))
    Rcpp::stop("All tree-depth alpha parameters must lie strictly between 0 and 1.");
  if (!std::isfinite(beta_mu) || beta_mu < 0.0 ||
      !std::isfinite(beta_mu_oc) || beta_mu_oc < 0.0 ||
      !std::isfinite(beta_tau) || beta_tau < 0.0 ||
      !std::isfinite(beta_tau_oc) || beta_tau_oc < 0.0)
    Rcpp::stop("All tree-depth beta parameters must be finite and non-negative.");
  if (!(tau_mu > 0.0) || !(tau_mu_oc > 0.0) ||
      !(tau_tau > 0.0) || !(tau_tau_oc > 0.0))
    Rcpp::stop("All terminal-node prior precisions must be positive.");
  if (!(nu > 0.0) || !(lambda > 0.0))
    Rcpp::stop("`nu` and `lambda` must be positive.");
  if (!std::isfinite(min_node_weight) || min_node_weight <= 0.0 ||
      !std::isfinite(min_active_weight) || min_active_weight < 0.0)
    Rcpp::stop("`min_node_weight` must be positive and `min_active_weight` non-negative.");
  if (max_cuts < 0)
    Rcpp::stop("`max_cuts` must be non-negative.");

  bcfusion::BcfInput in;
  in.n     = n;
  in.p     = X.ncol();
  in.p_tau = X_tau.ncol();

  in.X.resize(static_cast<std::size_t>(n) * in.p);
  for (int v = 0; v < in.p; v++)
    for (int i = 0; i < n; i++) {
      if (!std::isfinite(X(i, v))) Rcpp::stop("`X` must contain only finite values.");
      in.X[static_cast<std::size_t>(v) * n + i] = X(i, v);
    }

  in.X_tau.resize(static_cast<std::size_t>(n) * in.p_tau);
  for (int v = 0; v < in.p_tau; v++)
    for (int i = 0; i < n; i++) {
      if (!std::isfinite(X_tau(i, v)))
        Rcpp::stop("`X_tau` must contain only finite values.");
      in.X_tau[static_cast<std::size_t>(v) * n + i] = X_tau(i, v);
    }

  in.y.resize(n);
  in.d_oc.resize(n);
  in.z.resize(n);
  in.w.resize(n);
  for (int i = 0; i < n; i++) {
    if (!std::isfinite(y[i])) Rcpp::stop("`y` must contain only finite values.");
    if (!std::isfinite(D_oc[i]) || !(D_oc[i] == 0.0 || D_oc[i] == 1.0))
      Rcpp::stop("`D_oc` must be a binary 0/1 indicator.");
    if (!std::isfinite(Z[i]) || !(Z[i] == 0.0 || Z[i] == 1.0))
      Rcpp::stop("`Z` must be a binary 0/1 indicator.");
    if (D_oc[i] == 1.0 && power_weights[i] > 0.0 &&
        std::abs(power_weights[i] - 1.0) > 1e-12)
      Rcpp::stop("Positive-weight OC observations must have `power_weights = 1`.");
    in.y[i]         = y[i];
    in.d_oc[i] = D_oc[i];
    in.z[i]    = Z[i];
    in.w[i]         = power_weights[i];
  }

  in.alpha_mu      = alpha_mu;      in.beta_mu      = beta_mu;
  in.alpha_mu_oc   = alpha_mu_oc;   in.beta_mu_oc   = beta_mu_oc;
  in.alpha_tau     = alpha_tau;     in.beta_tau     = beta_tau;
  in.alpha_tau_oc  = alpha_tau_oc;  in.beta_tau_oc  = beta_tau_oc;
  in.tau_mu        = tau_mu;        in.tau_mu_oc   = tau_mu_oc;
  in.tau_tau       = tau_tau;       in.tau_tau_oc  = tau_tau_oc;
  in.nu            = nu;            in.lambda       = lambda;

  in.n_iter = n_iter;  in.n_burn = n_burn;  in.thin = thin;
  in.n_tree_mu      = n_tree_mu;
  in.n_tree_mu_oc   = n_tree_mu_oc;
  in.n_tree_tau     = n_tree_tau;
  in.n_tree_tau_oc  = n_tree_tau_oc;
  in.min_node_weight   = min_node_weight;
  in.min_active_weight = min_active_weight;
  in.max_cuts       = max_cuts;
  in.renormalise_likelihood = renormalise_likelihood;
  in.store_draws    = store_draws;
  in.return_trees   = return_trees;

  if (verbose)
    Rcpp::Rcout << "Running " << n_iter << " iterations (" << n_burn
                << " burn-in, thin " << thin << ")...\n";

  bcfusion::BcfOutput res = bcfusion::run_bcf(in, &bcf_interrupt);

  if (verbose) Rcpp::Rcout << "Done.\n";

  const int K = res.n_kept;

  NumericMatrix pm(store_draws ? n : 0, store_draws ? K : 0);
  NumericMatrix pm_oc(store_draws ? n : 0, store_draws ? K : 0);
  NumericMatrix pt(store_draws ? n : 0, store_draws ? K : 0);
  NumericMatrix pt_oc(store_draws ? n : 0, store_draws ? K : 0);
  if (store_draws) {
    for (int k = 0; k < K; k++)
      for (int i = 0; i < n; i++) {
        const std::size_t off = static_cast<std::size_t>(k) * n + i;
        pm(i, k)  = res.preds_mu[off];
        pm_oc(i, k) = res.preds_mu_oc[off];
        pt(i, k)  = res.preds_tau[off];
        pt_oc(i, k) = res.preds_tau_oc[off];
      }
  }

  NumericVector error_precision(K), error_sd(K);
  for (int k = 0; k < K; k++) {
    error_precision[k] = res.error_precisions[k];
    error_sd[k] = res.error_sds[k];
  }

  NumericVector mu_hat(n), mu_oc_hat(n), tau_hat(n), tau_oc_hat(n);
  NumericVector cate_uc_hat(n), cate_oc_hat(n);
  for (int i = 0; i < n; i++) {
    mu_hat[i]      = res.mu_hat[i];
    mu_oc_hat[i]   = res.mu_oc_hat[i];
    tau_hat[i]     = res.tau_hat[i];
    tau_oc_hat[i]  = res.tau_oc_hat[i];
    cate_uc_hat[i] = res.tau_hat[i];
    cate_oc_hat[i] = res.tau_hat[i] + res.tau_oc_hat[i];
  }

  NumericMatrix accept(4, 4);
  for (int f = 0; f < 4; f++)
    for (int m = 0; m < 4; m++) {
      const long long pnum = res.prop[f * 4 + m];
      accept(f, m) = (pnum > 0)
        ? static_cast<double>(res.acc[f * 4 + m]) / static_cast<double>(pnum)
        : NA_REAL;
    }
  rownames(accept) = CharacterVector::create("mu", "mu_oc_deviation", "tau", "tau_oc_deviation");
  colnames(accept) = CharacterVector::create("grow", "prune", "change", "swap");

  List final_trees;
  if (return_trees) {
    CharacterVector x_names = colnames(X);
    CharacterVector x_tau_names = colnames(X_tau);
    if (static_cast<int>(x_names.size()) != X.ncol()) {
      x_names = CharacterVector(X.ncol());
      for (int j = 0; j < X.ncol(); j++) x_names[j] = "X" + std::to_string(j + 1);
    }
    if (static_cast<int>(x_tau_names.size()) != X_tau.ncol()) {
      x_tau_names = CharacterVector(X_tau.ncol());
      for (int j = 0; j < X_tau.ncol(); j++)
        x_tau_names[j] = "X_tau" + std::to_string(j + 1);
    }
    final_trees = List::create(
      Named("mu") = serialize_forest(res.final_mu, x_names, res.y_sd),
      Named("mu_oc_deviation") = serialize_forest(res.final_mu_oc, x_names, res.y_sd),
      Named("tau") = serialize_forest(res.final_tau, x_tau_names, res.y_sd),
      Named("tau_oc_deviation") = serialize_forest(res.final_tau_oc, x_tau_names, res.y_sd)
    );
  }

  return List::create(
    Named("predictions_mu")      = pm,
    Named("predictions_mu_oc_deviation")  = pm_oc,
    Named("predictions_tau")     = pt,
    Named("predictions_tau_oc_deviation") = pt_oc,
    Named("mu_hat")              = mu_hat,
    Named("mu_oc_deviation_hat")          = mu_oc_hat,
    Named("tau_hat")             = tau_hat,
    Named("tau_oc_deviation_hat")         = tau_oc_hat,
    Named("cate_uc_hat")         = cate_uc_hat,
    Named("cate_oc_hat")         = cate_oc_hat,
    Named("error_precision")     = error_precision,
    Named("error_sd")            = error_sd,
    Named("acceptance")          = accept,
    Named("final_trees")         = final_trees,
    Named("n_kept")              = K,
    Named("y_mean")              = res.y_mean,
    Named("y_sd")                = res.y_sd,
    Named("likelihood_convention") = renormalise_likelihood
      ? "renormalised Gaussian power likelihood"
      : "unnormalised power likelihood"
  );
}
