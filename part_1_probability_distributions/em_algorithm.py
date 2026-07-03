import numpy as np
import pandas as pd
import os

# data
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "GaltonFamilies.csv")
df = pd.read_csv(csv_path)

fathers = df.drop_duplicates(subset="family")["father"].values.astype(float)
children = df["childHeight"].values.astype(float)

# remove labels by adding both arrays
data = np.concatenate([fathers, children])   
N = len(data)

SEP = "=" * 70
print(SEP)
print("  GALTON HEIGHTS  --  EM GAUSSIAN MIXTURE ANALYSIS")
print("  Populations : Fathers (Parents)  vs  Children")
print(SEP)
print(f"  Fathers : n={len(fathers):4d}  mean={fathers.mean():.4f}  "
      f"std={fathers.std():.4f}")
print(f"  Children : n={len(children):4d}  mean={children.mean():.4f}  "
      f"std={children.std():.4f}")
print(f"  Combined (mixed): n={N:4d}  mean={data.mean():.4f}  "
      f"std={data.std():.4f}")
print(SEP)



#  GMM functions
def gauss(x, mu, var):
    """
    Gaussian PDF   N(x ; mu, var)   vectorised over x.

        f(x) = exp( -(x-mu)^2 / (2*var) )  /  sqrt(2*pi*var)
    """
    var = max(float(var), 1e-10)
    return np.exp(-0.5 * (x - mu) ** 2 / var) / np.sqrt(2.0 * np.pi * var)


def log_likelihood(x, pi1, pi2, mu1, mu2, v1, v2):
    """
    Observed-data log-likelihood of the 2-component GMM:

        L(theta) = sum_i  log[ pi1*N(x_i;mu1,v1) + pi2*N(x_i;mu2,v2) ]

    1e-300 guards against log(0).
    """
    mix = pi1 * gauss(x, mu1, v1) + pi2 * gauss(x, mu2, v2)
    return float(np.sum(np.log(mix + 1e-300)))


def e_step(x, pi1, pi2, mu1, mu2, v1, v2):
    """
    r_k[i] = P(z_i = k | x_i, theta)
                    pi_k  *  N(x_i ; mu_k, v_k)
            = -----------------------------------------------
                pi_1*N(x_i;mu1,v1)  +  pi_2*N(x_i;mu2,v2)
    """
    p1 = pi1 * gauss(x, mu1, v1)
    p2 = pi2 * gauss(x, mu2, v2)
    d = p1 + p2 + 1e-300
    return p1 / d, p2 / d


def m_step(x, r1, r2):
    """
    N_k  = sum_i r_k[i]                          (effective count)
    mu_k = (sum_i r_k[i]*x_i) / N_k             (weighted mean)
    v_k  = (sum_i r_k[i]*(x_i-mu_k)^2) / N_k   (weighted variance)
    pi_k = N_k / N                               (mixing weight)
    """
    n = len(x)
    s1 = r1.sum()
    s2 = r2.sum()
    mu1 = (r1 * x).sum() / s1
    mu2 = (r2 * x).sum() / s2
    v1 = (r1 * (x - mu1) ** 2).sum() / s1
    v2 = (r2 * (x - mu2) ** 2).sum() / s2
    return mu1, mu2, v1, v2, s1 / n, s2 / n


# k-means initialization
def kmeans2(x, seed=42):
    """
    Seeded at the 25th and 75th percentiles for stability.
    """
    c1 = float(np.percentile(x, 25))
    c2 = float(np.percentile(x, 75))

    for _ in range(200):
        mask = np.abs(x - c1) <= np.abs(x - c2)
        new_c1 = x[mask].mean()
        new_c2 = x[~mask].mean()
        if abs(new_c1 - c1) + abs(new_c2 - c2) < 1e-10:
            break
        c1, c2 = new_c1, new_c2

    # Ensure c1 < c2 (lower cluster = Component 1)
    if c1 > c2:
        c1, c2 = c2, c1
        mask = ~mask

    return c1, c2, mask


# EM loop
def run_em(x, max_iter=200, tol=1e-4):
    """
    Full EM algorithm for a 2-component 1-D Gaussian Mixture Model.
    """

    # -- K-means seed ---------------------------------------------------------
    c1, c2, mask = kmeans2(x)
    mu1, mu2 = c1, c2
    v1 = x[mask].var()
    v2 = x[~mask].var()
    pi1 = mask.mean()
    pi2 = (~mask).mean()
    ll = log_likelihood(x, pi1, pi2, mu1, mu2, v1, v2)

    print(f"\n  K-means init: mu1={mu1:.4f}  mu2={mu2:.4f}  "
          f"pi1={pi1:.4f}  pi2={pi2:.4f}")

    history = [{"it": 0, "mu1": mu1, "mu2": mu2,
                "v1": v1, "v2": v2, "pi1": pi1, "pi2": pi2, "ll": ll}]

    # -- Iterate --------------------------------------------------------------
    for i in range(1, max_iter + 1):

        # E-Step
        r1, r2 = e_step(x, pi1, pi2, mu1, mu2, v1, v2)

        # M-Step
        mu1, mu2, v1, v2, pi1, pi2 = m_step(x, r1, r2)

        # Keep Component 1 = lower-mean group
        if mu1 > mu2:
            mu1, mu2 = mu2, mu1
            v1, v2 = v2, v1
            pi1, pi2 = pi2, pi1

        ll_new = log_likelihood(x, pi1, pi2, mu1, mu2, v1, v2)
        history.append({"it": i, "mu1": mu1, "mu2": mu2,
                        "v1": v1, "v2": v2, "pi1": pi1, "pi2": pi2, "ll": ll_new})

        delta = abs(ll_new - ll)
        if delta < tol:
            print(f"  EM converged at iteration {i}  "
                  f"(|delta LL| = {delta:.3e}  <  tol = {tol:.0e})")
            break
        ll = ll_new
    else:
        print(f"  EM reached max_iter={max_iter} without full convergence.")

    return history


print()
history = run_em(data)


# tracking table
W = 111
print()
print("=" * W)
print("  EM ALGORITHM -- TRACKING TABLE")
print("=" * W)
print(f"  {'Iter':>4}  "
      f"{'mu1 (Children)':>15}  "
      f"{'mu2 (Parents)':>19}  "
      f"{'variance_1':>10}  "
      f"{'variance_2':>10}  "
      f"{'pi1':>8}  "
      f"{'pi2':>8}  "
      f"{'Log-Likelihood':>16}")
print("-" * W)

for row in history[:3]:
    print(f"  {row['it']:>4}  "
          f"{row['mu1']:>15.4f}  "
          f"{row['mu2']:>19.4f}  "
          f"{row['v1']:>10.4f}  "
          f"{row['v2']:>10.4f}  "
          f"{row['pi1']:>8.4f}  "
          f"{row['pi2']:>8.4f}  "
          f"{row['ll']:>16.4f}")

print("=" * W)


# summary

final = history[-1]
MU1, MU2 = final["mu1"], final["mu2"]
V1, V2 = final["v1"], final["v2"]
PI1, PI2 = final["pi1"], final["pi2"]

print(f"\n  CONVERGED PARAMETERS  (after {final['it']} EM iteration(s))")
print(f"\n  {'':36}  {'mu (in)':>10}  {'variance':>8}  {'variance_2':>10}  {'pi':>8}")
print(f"  {'-' * 72}")
print(f"  {'Component 1 -- Children (Child)':36}  "
      f"{MU1:>10.4f}  {np.sqrt(V1):>8.4f}  {V1:>10.4f}  {PI1:>8.4f}")
print(f"  {'Component 2 -- Fathers (Parent)':36}  "
      f"{MU2:>10.4f}  {np.sqrt(V2):>8.4f}  {V2:>10.4f}  {PI2:>8.4f}")

print(f"\n  Ground-truth reference (true labels):")
print(f"    Children mean={children.mean():.4f}  "
      f"std={children.std():.4f}  "
      f"(EM mean error: {abs(MU1-children.mean()):.4f} in)")
print(f"    Fathers mean={fathers.mean():.4f}  "
      f"std={fathers.std():.4f}  "
      f"(EM mean error: {abs(MU2-fathers.mean()):.4f} in)")


# classification function

def classify_height(height_in):
    """
    Bayesian soft classification using the converged GMM.

    Applies Bayes' theorem:

        P(Child  | h) = pi1 * N(h; mu1, v1)
                        ----------------------------------------
                        pi1*N(h;mu1,v1)  +  pi2*N(h;mu2,v2)

        P(Parent | h) = pi2 * N(h; mu2, v2)  /  (same denominator)

    Parameters
    ----------
    height_in : float  --  height in inches

    Returns
    -------
    p_child  : float  --  P(Child  | height)
    p_parent : float  --  P(Parent | height)
    """
    wc = PI1 * gauss(np.array([height_in]), MU1, V1)[0]
    wp = PI2 * gauss(np.array([height_in]), MU2, V2)[0]
    t = wc + wp + 1e-300
    return wc / t, wp / t


# enter height
print()
print(SEP)
print("  Enter a height in inches.  Type  q  to quit.\n")

while True:
    try:
        raw = input("  Height (in) > ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n  Exiting.")
        break

    if raw.lower() in ("q", "quit", "exit", ""):
        print("  Exiting.")
        break

    try:
        h = float(raw)
        pc, pp = classify_height(h)
        label = "CHILD" if pc >= pp else "PARENT (father)"
        conf = max(pc, pp) * 100
        print(f"\n    Height         : {h:.2f} in  ({h*2.54:.1f} cm)")
        print(f"    P(Child)       : {pc:.6f}   ({pc*100:6.2f} %)")
        print(f"    P(Parent)      : {pp:.6f}   ({pp*100:6.2f} %)")
        print(f"    Classification : {label}   [{conf:.1f} % confidence]\n")
    except ValueError:
        print("  Please enter a valid number  (e.g.  67.5  or  72).\n")