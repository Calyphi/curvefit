import { useState, useEffect, useMemo, useCallback, useRef } from "react";
import { Scatter, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ComposedChart } from "recharts";
import ProModal from "./ProModal";

// ============================================================
// SAFE MATH
// ============================================================
const safeExp = (x) => Math.exp(Math.max(-700, Math.min(700, x)));

// Safe power: prevents Infinity from x^n with large n
const safePow = (base, exp) => {
  if (base === 0) return exp > 0 ? 0 : NaN;
  if (!isFinite(exp) || !isFinite(base)) return NaN;
  const absBase = Math.abs(base);
  // If |exp * log(|base|)| > 700, result would overflow/underflow
  if (absBase > 0 && Math.abs(exp * Math.log(absBase)) > 700) return exp > 0 ? (absBase > 1 ? Infinity : 0) : (absBase > 1 ? 0 : Infinity);
  return Math.pow(base, exp);
};

// Track clamp usage per fit
function createSafeExpTracker() {
  let count = 0;
  const tracked = (x) => {
    const clamped = Math.max(-700, Math.min(700, x));
    if (clamped !== x) count++;
    return Math.exp(clamped);
  };
  tracked.getCount = () => count;
  tracked.reset = () => { count = 0; };
  return tracked;
}

// ============================================================
// LEVENBERG-MARQUARDT v3
// + Covariance matrix → standard errors → confidence intervals
// + Fit warnings
// ============================================================
function levenbergMarquardt(func, xData, yData, initialParams, options = {}) {
  const {
    maxIter = 200,
    tolerance = 1e-7,
    lambdaInit = 0.01,
    lambdaUp = 10,
    lambdaDown = 0.1,
    positiveIdx = [],
  } = options;

  const n = xData.length;
  const p = initialParams.length;
  const warnings = [];
  const tracker = createSafeExpTracker();

  // Reparametrize positive params via log transform
  const posSet = new Set(positiveIdx); // FIX: Set instead of includes()
  const toInternal = (params) => params.map((v, i) =>
    posSet.has(i) ? Math.log(Math.abs(v) + 1e-30) : v
  );
  let paramClampCount = 0;
  const toExternal = (internal) => internal.map((v, i) => {
    if (!posSet.has(i)) return v;
    const clamped = Math.max(-500, Math.min(500, v));
    if (clamped !== v) paramClampCount++;
    return Math.exp(clamped);
  });

  // Wrap func with reparametrization + exp tracking
  const wrappedFunc = (x, internalParams) => {
    const ext = toExternal(internalParams);
    return func(x, ext, tracker);
  };

  let params = toInternal(initialParams);
  let lambda = lambdaInit;
  let stagnantCount = 0;

  const computeResiduals = (par) => {
    const res = new Array(n);
    for (let i = 0; i < n; i++) {
      const val = wrappedFunc(xData[i], par);
      if (isFinite(val)) {
        res[i] = yData[i] - val;
      } else {
        res[i] = NaN; // Mark invalid, don't fake it
      }
    }
    return res;
  };

  // Cost ignoring NaN residuals
  const computeCost = (residuals) => {
    let s = 0, count = 0;
    for (let i = 0; i < residuals.length; i++) {
      if (isFinite(residuals[i])) { s += residuals[i] * residuals[i]; count++; }
    }
    return { cost: s, validCount: count };
  };

  // Jacobian with relative eps
  const computeJacobian = (par) => {
    const J = [];
    for (let i = 0; i < n; i++) {
      J[i] = new Array(p);
      for (let j = 0; j < p; j++) {
        const epsJ = 1e-6 * (Math.abs(par[j]) + 1);
        const p1 = [...par]; p1[j] += epsJ;
        const p2 = [...par]; p2[j] -= epsJ;
        const f1 = wrappedFunc(xData[i], p1);
        const f2 = wrappedFunc(xData[i], p2);
        J[i][j] = (isFinite(f1) && isFinite(f2)) ? (f1 - f2) / (2 * epsJ) : 0;
      }
    }
    return J;
  };

  let residuals = computeResiduals(params);
  let { cost, validCount } = computeCost(residuals);

  for (let iter = 0; iter < maxIter; iter++) {
    const J = computeJacobian(params);

    const JtJ = Array.from({ length: p }, () => new Array(p).fill(0));
    const JtR = new Array(p).fill(0);

    for (let i = 0; i < n; i++) {
      if (!isFinite(residuals[i])) continue; // Skip NaN rows
      for (let j = 0; j < p; j++) {
        JtR[j] += J[i][j] * residuals[i];
        for (let k = 0; k < p; k++) {
          JtJ[j][k] += J[i][j] * J[i][k];
        }
      }
    }

    const A = JtJ.map((row, i) => row.map((v, j) =>
      i === j ? v + lambda * (Math.abs(v) + 1e-8) : v
    ));

    const delta = solveLinear(A, JtR);
    if (!delta || delta.some(d => !isFinite(d))) {
      lambda *= lambdaUp;
      stagnantCount++;
      if (stagnantCount > 20) break;
      continue;
    }

    // Relative stopping
    const paramNorm = Math.sqrt(params.reduce((s, v) => s + v * v, 0));
    const deltaNorm = Math.sqrt(delta.reduce((s, v) => s + v * v, 0));
    if (deltaNorm / (paramNorm + 1) < tolerance) break;

    const newParams = params.map((v, i) => v + delta[i]);
    const newResiduals = computeResiduals(newParams);
    const { cost: newCost } = computeCost(newResiduals);

    if (isFinite(newCost) && newCost < cost) {
      if (Math.abs(cost - newCost) / (cost + 1e-30) < 1e-12) {
        stagnantCount++;
        if (stagnantCount > 10) break;
      } else {
        stagnantCount = 0;
      }
      params = newParams;
      residuals = newResiduals;
      cost = newCost;
      lambda *= lambdaDown;
    } else {
      lambda *= lambdaUp;
      stagnantCount++;
      if (stagnantCount > 20) break;
    }
    if (lambda > 1e16) break;
  }

  const externalParams = toExternal(params);

  // ---- METRICS (FIX: R² can be negative) ----
  const yMean = yData.reduce((s, y) => s + y, 0) / n;
  const ssTot = yData.reduce((s, y) => s + (y - yMean) ** 2, 0);
  const ssRes = cost;
  const rSquared = ssTot > 0 ? 1 - ssRes / ssTot : 0; // NO Math.max(0,...) — can be negative
  const adjR2 = validCount > p + 1 ? 1 - (1 - rSquared) * (validCount - 1) / (validCount - p - 1) : rSquared;

  // AICc (falls back to AIC when n too small for correction)
  const aic = validCount > 0 ? validCount * Math.log(ssRes / validCount + 1e-30) + 2 * p : Infinity;
  const aicc = (validCount > p + 1) ? aic + (2 * p * (p + 1)) / (validCount - p - 1) : aic; // Fallback to AIC, not Infinity
  const aiccFallback = !(validCount > p + 1); // true when n too small for AICc correction

  // ---- CONFIDENCE INTERVALS via covariance matrix ----
  // Cov ≈ s² * (JᵀJ)⁻¹ where s² = SSR / (n - p)
  let stdErrors = null;
  let ci95 = null;
  let covMatrix = null;
  let s2 = null;
  const dof = validCount - p;

  // t-critical values for 95% CI (two-tailed, α=0.025 each tail)
  // Exact values for DOF 1–30, normal approximation for DOF > 30
  const T_TABLE_95 = [0, 12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
    2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
    2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042];
  const tVal = dof >= 1 && dof <= 30 ? T_TABLE_95[dof] : dof > 30 ? 1.96 : 2;

  if (dof > 0) {
    s2 = ssRes / dof;
    // Recompute JtJ at final params — ONLY over valid points
    const Jfinal = [];
    for (let i = 0; i < n; i++) {
      // Check if this point produces a finite prediction
      const baseVal = func(xData[i], externalParams, safeExp);
      if (!isFinite(baseVal)) continue; // Skip invalid points — same as LM

      const row = new Array(p);
      for (let j = 0; j < p; j++) {
        const epsJ = 1e-6 * (Math.abs(externalParams[j]) + 1);
        const pp = [...externalParams]; pp[j] += epsJ;
        const pm = [...externalParams]; pm[j] -= epsJ;
        const f1 = func(xData[i], pp, safeExp);
        const f2 = func(xData[i], pm, safeExp);
        row[j] = (isFinite(f1) && isFinite(f2)) ? (f1 - f2) / (2 * epsJ) : 0;
      }
      Jfinal.push(row);
    }

    const JtJfinal = Array.from({ length: p }, () => new Array(p).fill(0));
    for (let i = 0; i < Jfinal.length; i++) {
      for (let j = 0; j < p; j++) {
        for (let k = 0; k < p; k++) {
          JtJfinal[j][k] += Jfinal[i][j] * Jfinal[i][k];
        }
      }
    }

    // Tikhonov regularization: add ε·I to stabilize near-singular JᵀJ
    // Use max(diag) for scale-invariance (more robust than trace/p)
    let maxDiag = 0;
    for (let j = 0; j < p; j++) maxDiag = Math.max(maxDiag, Math.abs(JtJfinal[j][j]));
    const ridge = 1e-12 * (maxDiag || 1);
    let ridgeApplied = false;
    for (let j = 0; j < p; j++) {
      if (JtJfinal[j][j] < ridge * 100) ridgeApplied = true;
      JtJfinal[j][j] += ridge;
    }

    covMatrix = invertMatrix(JtJfinal);
    if (covMatrix) {
      stdErrors = new Array(p);
      ci95 = new Array(p);
      let allValid = true;
      for (let j = 0; j < p; j++) {
        const variance = s2 * covMatrix[j][j];
        if (variance > 0 && isFinite(variance)) {
          stdErrors[j] = Math.sqrt(variance);
          ci95[j] = [externalParams[j] - tVal * stdErrors[j], externalParams[j] + tVal * stdErrors[j]];
        } else {
          stdErrors[j] = NaN;
          ci95[j] = [NaN, NaN];
          allValid = false;
        }
      }
      if (!allValid) warnings.push("Some parameter uncertainties could not be estimated (singular covariance).");
      if (ridgeApplied) warnings.push("Covariance regularized — uncertainties are approximate (near-singular JᵀJ).");
    } else {
      warnings.push("Covariance matrix is singular — parameter uncertainties unavailable.");
    }

    // Check parameter correlations
    if (covMatrix) {
      for (let j = 0; j < p; j++) {
        for (let k = j + 1; k < p; k++) {
          const denom = Math.sqrt(Math.abs(covMatrix[j][j] * covMatrix[k][k]));
          if (denom > 0) {
            const corr = Math.abs(covMatrix[j][k] / denom);
            if (corr > 0.95) {
              warnings.push(`High correlation (|r|=${corr.toFixed(3)}) between parameters — model may be overparameterized.`);
              break;
            }
          }
        }
        if (warnings.length > 3) break;
      }
    }
  } else {
    warnings.push(`Too few data points (n=${validCount}) for ${p} parameters — no degrees of freedom.`);
  }

  // ---- FIT WARNINGS ----
  const clampCount = tracker.getCount();
  if (clampCount > 0) {
    warnings.push(`Exp overflow clamped ${clampCount} times — model may be inappropriate for this data range.`);
  }
  if (paramClampCount > 0) {
    warnings.push(`Parameter reparametrization clamped ${paramClampCount} times — positive parameters may be at bounds.`);
  }

  if (rSquared < 0) {
    warnings.push("Negative R²: this model fits worse than a horizontal line at the mean.");
  }
  if (aiccFallback) {
    warnings.push("AICc correction unavailable (low data) — using AIC. Rankings less reliable.");
  }

  // FIX: finalResiduals with NaN for invalid points, never 0
  // CRITICAL: Revalidate with FINAL params (may differ from mid-optimization validCount)
  const finalResiduals = xData.map((x, i) => {
    const val = func(x, externalParams, safeExp);
    return isFinite(val) ? yData[i] - val : NaN;
  });

  const finalInvalidCount = finalResiduals.filter(r => !isFinite(r)).length;
  if (finalInvalidCount > 0) {
    warnings.push(`${finalInvalidCount} data point(s) produced non-finite predictions — excluded from metrics.`);
  }

  return {
    params: externalParams, rSquared, adjR2, aic, aicc, aiccFallback, cost,
    residuals: finalResiduals, stdErrors, ci95, warnings, dof,
    validCount, covMatrix, s2, tVal, nParams: p, finalInvalidCount
  };
}

// ============================================================
// MATRIX INVERSE (for covariance)
// ============================================================
function invertMatrix(M) {
  const n = M.length;
  const aug = M.map((row, i) => {
    const r = [...row];
    for (let j = 0; j < n; j++) r.push(i === j ? 1 : 0);
    return r;
  });

  for (let col = 0; col < n; col++) {
    let maxRow = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(aug[row][col]) > Math.abs(aug[maxRow][col])) maxRow = row;
    }
    [aug[col], aug[maxRow]] = [aug[maxRow], aug[col]];
    if (Math.abs(aug[col][col]) < 1e-20) return null;

    const pivot = aug[col][col];
    for (let j = 0; j < 2 * n; j++) aug[col][j] /= pivot;

    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const f = aug[row][col];
      for (let j = 0; j < 2 * n; j++) aug[row][j] -= f * aug[col][j];
    }
  }

  return aug.map(row => row.slice(n));
}

function solveLinear(A, b) {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let maxRow = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(M[row][col]) > Math.abs(M[maxRow][col])) maxRow = row;
    }
    [M[col], M[maxRow]] = [M[maxRow], M[col]];
    if (Math.abs(M[col][col]) < 1e-20) return null;
    for (let row = col + 1; row < n; row++) {
      const f = M[row][col] / M[col][col];
      for (let j = col; j <= n; j++) M[row][j] -= f * M[col][j];
    }
  }
  const x = new Array(n);
  for (let i = n - 1; i >= 0; i--) {
    x[i] = M[i][n];
    for (let j = i + 1; j < n; j++) x[i] -= M[i][j] * x[j];
    x[i] /= M[i][i];
  }
  return x;
}

// ============================================================
// SCIENTIFIC MODELS v3 — with family field for dedup
// ============================================================
function buildModels(xData, yData) {
  const n = xData.length;
  if (n < 3) return [];
  const xMin = Math.min(...xData), xMax = Math.max(...xData);
  const yMin = Math.min(...yData), yMax = Math.max(...yData);
  const yMean = yData.reduce((s, y) => s + y, 0) / n;
  const xMean = xData.reduce((s, x) => s + x, 0) / n;
  const yRange = yMax - yMin || 1;
  const xRange = xMax - xMin || 1;

  const linReg = (xs, ys) => {
    const nn = xs.length;
    if (nn < 2) return { a: 0, b: ys[0] || 0 };
    const mx = xs.reduce((s, x) => s + x, 0) / nn;
    const my = ys.reduce((s, y) => s + y, 0) / nn;
    let num = 0, den = 0;
    for (let i = 0; i < nn; i++) { num += (xs[i] - mx) * (ys[i] - my); den += (xs[i] - mx) ** 2; }
    return { a: den > 1e-30 ? num / den : 0, b: my - (den > 1e-30 ? num / den : 0) * mx };
  };

  const xAtY = (target) => {
    let best = xMean, bestDist = Infinity;
    for (let i = 0; i < n; i++) {
      const d = Math.abs(yData[i] - target);
      if (d < bestDist) { bestDist = d; best = xData[i]; }
    }
    return best || xMean;
  };

  const iMax = yData.indexOf(yMax);
  const posPairs = [];
  for (let i = 0; i < n; i++) {
    if (xData[i] > 0 && yData[i] > 0) posPairs.push({ x: xData[i], y: yData[i], lx: Math.log(xData[i]), ly: Math.log(yData[i]) });
  }
  const posXCount = xData.filter(x => x > 0).length;
  const posYCount = yData.filter(y => y > 0).length;

  const estimateFrequency = () => {
    let crossings = 0;
    for (let i = 1; i < n; i++) {
      if ((yData[i] - yMean) * (yData[i - 1] - yMean) < 0) crossings++;
    }
    return 2 * Math.PI * Math.max(crossings / 2, 0.5) / xRange;
  };

  const models = [];
  const lr = linReg(xData, yData);

  // All funcs now accept (x, p, exp) where exp is safeExp or tracker
  const add = (family, name, eq, nP, func, init, pNames, posIdx) => {
    models.push({ family, name, equation: eq, nParams: nP, func, init, paramNames: pNames, positiveIdx: posIdx });
  };

  // Polynomial
  add("Linear", "Linear", "y = a·x + b", 2,
    (x, p) => p[0] * x + p[1], [lr.a, lr.b], ["a", "b"], []);
  add("Quadratic", "Quadratic", "y = a·x² + b·x + c", 3,
    (x, p) => p[0] * x * x + p[1] * x + p[2], [0, lr.a, lr.b], ["a", "b", "c"], []);
  add("Cubic", "Cubic", "y = a·x³ + b·x² + c·x + d", 4,
    (x, p) => p[0] * x ** 3 + p[1] * x ** 2 + p[2] * x + p[3], [0, 0, lr.a, lr.b], ["a", "b", "c", "d"], []);

  // Exponential
  if (posYCount > n * 0.7) {
    const logY = yData.map(y => y > 0 ? Math.log(y) : 0);
    const elr = linReg(xData, logY);
    add("ExpGrowth", "Exponential Growth", "y = a·exp(b·x)", 2,
      (x, p, E) => p[0] * (E || safeExp)(p[1] * x),
      [Math.exp(elr.b), elr.a], ["a", "b"], [0]);
  }
  add("ExpDecay", "Exponential Decay", "y = a·exp(−b·x) + c", 3,
    (x, p, E) => p[0] * (E || safeExp)(-p[1] * x) + p[2],
    [yRange, 1 / xRange, yMin], ["a", "b", "c"], [0, 1]);

  // Power / Log
  if (posPairs.length > n * 0.7) {
    const plr = linReg(posPairs.map(q => q.lx), posPairs.map(q => q.ly));
    add("PowerLaw", "Power Law", "y = a·x^b", 2,
      (x, p) => { const v = safePow(Math.abs(x) + 1e-30, p[1]); return isFinite(v) ? p[0] * v : NaN; },
      [Math.exp(plr.b), plr.a], ["a", "b"], [0]);
  }
  if (posXCount > n * 0.8) {
    const logPairs = [];
    for (let i = 0; i < n; i++) if (xData[i] > 0) logPairs.push({ lx: Math.log(xData[i]), y: yData[i] });
    if (logPairs.length > 2) {
      const llr = linReg(logPairs.map(q => q.lx), logPairs.map(q => q.y));
      add("Logarithmic", "Logarithmic", "y = a·ln(x) + b", 2,
        (x, p) => p[0] * Math.log(Math.abs(x) + 1e-30) + p[1],
        [llr.a, llr.b], ["a", "b"], []);
    }
  }

  // Logistic (multi-start)
  const logSeeds = [
    [yMax, 4 / xRange, xMean],
    [yRange, 2 / xRange, xAtY((yMax + yMin) / 2)],
    [yMax * 1.2, 8 / xRange, xMean + xRange * 0.1],
  ];
  logSeeds.forEach((seed, si) => add("Logistic", si === 0 ? "Logistic" : `Logistic (seed ${si + 1})`,
    "y = L / (1 + exp(−k·(x − x₀)))", 3,
    (x, p, E) => p[0] / (1 + (E || safeExp)(-p[1] * (x - p[2]))),
    seed, ["L", "k", "x₀"], [0, 1]));

  // Gaussian (multi-start)
  const gSeeds = [
    [yRange, xData[iMax] || xMean, xRange / 4, yMin],
    [yRange * 0.8, xMean, xRange / 6, yMin],
    [yRange, xData[iMax] || xMean, xRange / 2, (yMin + yMean) / 2],
  ];
  gSeeds.forEach((seed, si) => add("Gaussian", si === 0 ? "Gaussian" : `Gaussian (seed ${si + 1})`,
    "y = a·exp(−(x−μ)²/(2σ²)) + c", 4,
    (x, p, E) => p[0] * (E || safeExp)(-((x - p[1]) ** 2) / (2 * p[2] ** 2 + 1e-30)) + p[3],
    seed, ["a", "μ", "σ", "c"], [0, 2]));

  // Enzyme kinetics
  if (posXCount > n * 0.7) {
    add("MichaelisMenten", "Michaelis-Menten", "y = Vmax·x / (Km + x)", 2,
      (x, p) => p[0] * x / (p[1] + x + 1e-30),
      [yMax * 1.2, xAtY(yMax / 2)], ["Vmax", "Km"], [0, 1]);
    add("Hill", "Hill Equation", "y = Vmax·xⁿ / (Kⁿ + xⁿ)", 3,
      (x, p) => { const xn = safePow(Math.abs(x) + 1e-30, p[2]); const kn = safePow(p[1] + 1e-30, p[2]); return isFinite(xn) && isFinite(kn) ? p[0] * xn / (kn + xn + 1e-30) : NaN; },
      [yMax * 1.2, xAtY(yMax / 2), 1.5], ["Vmax", "K", "n"], [0, 1, 2]);
  }

  // 4PL Dose-Response
  if (posXCount > n * 0.5) {
    add("4PL", "4PL Dose-Response", "y = d + (a−d)/(1+(x/c)ᵇ)", 4,
      (x, p) => { const pw = safePow(Math.abs(x) / (p[2] + 1e-30), p[1]); return isFinite(pw) ? p[3] + (p[0] - p[3]) / (1 + pw) : NaN; },
      [yMax, 1, xAtY((yMax + yMin) / 2), yMin], ["a", "b", "c", "d"], [1, 2]);
  }

  // Adsorption
  if (posXCount > n * 0.7) {
    add("Langmuir", "Langmuir Isotherm", "y = qmax·KL·x / (1 + KL·x)", 2,
      (x, p) => p[0] * p[1] * x / (1 + p[1] * x + 1e-30),
      [yMax, 1 / (xAtY(yMax / 2) + 1e-10)], ["qmax", "KL"], [0, 1]);
  }
  if (posPairs.length > n * 0.7) {
    add("Freundlich", "Freundlich Isotherm", "y = Kf·x^(1/n)", 2,
      (x, p) => { const v = safePow(Math.abs(x) + 1e-30, 1 / (p[1] + 1e-30)); return isFinite(v) ? p[0] * v : NaN; },
      [yMean, 2], ["Kf", "n"], [0, 1]);
  }

  // Arrhenius: k = A·exp(−Ea/(R·T)) where R = 8.314 J/(mol·K), T in Kelvin
  const R_GAS = 8.314;
  if (posXCount > n * 0.8 && xMin > 0) {
    // Smart init: linearize ln(y) = ln(A) - Ea/(R·x), regress ln(y) vs 1/x
    const arrPairs = [];
    for (let i = 0; i < n; i++) {
      if (xData[i] > 0 && yData[i] > 0) arrPairs.push({ invX: 1 / xData[i], lnY: Math.log(yData[i]) });
    }
    let arrA = yMean, arrEa = 50000; // defaults: 50 kJ/mol
    if (arrPairs.length > 2) {
      const arrLR = linReg(arrPairs.map(q => q.invX), arrPairs.map(q => q.lnY));
      arrEa = Math.abs(arrLR.a * R_GAS); // slope = -Ea/R → Ea = -slope*R
      arrA = Math.exp(arrLR.b);
    }
    add("Arrhenius", "Arrhenius", "y = A·exp(−Ea/(R·T))", 2,
      (x, p, E) => p[0] * (E || safeExp)(-p[1] / (R_GAS * x + 1e-30)),
      [arrA, arrEa], ["A", "Ea (J/mol)"], [0, 1]);
  }

  // Stretched Exponential (multi-start)
  const kwwSeeds = [[yRange, xRange / 2, 0.7, yMin], [yRange, xRange / 5, 0.5, yMin], [yRange * 0.8, xRange, 1.0, (yMin + yMean) / 2]];
  kwwSeeds.forEach((seed, si) => add("KWW", si === 0 ? "Stretched Exponential" : `Stretched Exp (seed ${si + 1})`,
    "y = a·exp(−(x/τ)^β) + c", 4,
    (x, p, E) => { const pw = safePow(Math.abs(x) / (p[1] + 1e-30), p[2]); return isFinite(pw) ? p[0] * (E || safeExp)(-pw) + p[3] : NaN; },
    seed, ["a", "τ", "β", "c"], [0, 1, 2]));

  // Growth
  add("SatGrowth", "Saturation Growth", "y = a·(1 − exp(−b·x))", 2,
    (x, p, E) => p[0] * (1 - (E || safeExp)(-p[1] * x)),
    [yMax, 2 / xRange], ["a", "b"], [0, 1]);

  if (posXCount > n * 0.7) {
    add("Weibull", "Weibull CDF", "y = a·(1 − exp(−(x/λ)^k))", 3,
      (x, p, E) => { const pw = safePow(Math.abs(x) / (p[1] + 1e-30), p[2]); return isFinite(pw) ? p[0] * (1 - (E || safeExp)(-pw)) : NaN; },
      [yMax, xMean, 1.5], ["a", "λ", "k"], [0, 1, 2]);
  }

  // Reciprocal
  add("Reciprocal", "Reciprocal", "y = a/(x + b) + c", 3,
    (x, p) => p[0] / (x + p[1] + 1e-30) + p[2],
    [yRange * xRange, xMean, yMin], ["a", "b", "c"], []);

  // 5PL Dose-Response (asymmetric sigmoid)
  if (posXCount > n * 0.5) {
    add("5PL", "5PL Dose-Response", "y = d + (a−d)/((1+(x/c)ᵇ)ᵍ)", 5,
      (x, p) => { const pw = safePow(Math.abs(x) / (p[2] + 1e-30), p[1]); if (!isFinite(pw)) return NaN; const denom = safePow(1 + pw, p[4]); return isFinite(denom) && denom !== 0 ? p[3] + (p[0] - p[3]) / denom : NaN; },
      [yMax, 1, xAtY((yMax + yMin) / 2), yMin, 1], ["a", "b", "c", "d", "g"], [1, 2, 4]);
  }

  // Bi-Exponential Decay (5 params — needs sufficient data)
  if (n >= 8) {
    add("BiExp", "Bi-Exponential Decay", "y = a₁·exp(−k₁·x) + a₂·exp(−k₂·x) + c", 5,
      (x, p, E) => p[0] * (E || safeExp)(-p[1] * x) + p[2] * (E || safeExp)(-p[3] * x) + p[4],
      [yRange * 0.6, 2 / xRange, yRange * 0.3, 0.5 / xRange, yMin],
      ["a₁", "k₁", "a₂", "k₂", "c"], [0, 1, 2, 3]);
  }

  // Gompertz Growth: y = a*exp(-b*exp(-c*x))
  // Linearization: if y>0 and y<a, then ln(-ln(y/a)) = ln(b) - c*x
  {
    const aGuess = yMax * 1.1;
    let bGuess = 5, cGuess = 0.5 / (xRange + 1e-30);
    // Try linearization for better seeds
    const gomPairs = [];
    for (let gi = 0; gi < n; gi++) {
      if (yData[gi] > 0 && yData[gi] < aGuess) {
        const lnlnRatio = Math.log(-Math.log(yData[gi] / aGuess));
        if (isFinite(lnlnRatio)) gomPairs.push({ x: xData[gi], y: lnlnRatio });
      }
    }
    if (gomPairs.length > 2) {
      const glr = linReg(gomPairs.map(q => q.x), gomPairs.map(q => q.y));
      if (isFinite(glr.a) && isFinite(glr.b)) {
        cGuess = Math.abs(glr.a) || cGuess;
        bGuess = Math.exp(Math.max(-10, Math.min(10, glr.b)));
      }
    }
    add("Gompertz", "Gompertz", "y = a·exp(−b·exp(−c·x))", 3,
      (x, p, E) => p[0] * (E || safeExp)(-p[1] * (E || safeExp)(-p[2] * x)),
      [aGuess, bGuess, cGuess], ["a", "b", "c"], [0, 1, 2]);
  }

  // Lorentzian (Cauchy peak)
  add("Lorentzian", "Lorentzian", "y = a/((x−x₀)² + γ²) + c", 4,
    (x, p) => p[0] / ((x - p[1]) ** 2 + p[2] ** 2 + 1e-30) + p[3],
    [yRange * (xRange / 4) ** 2, xData[iMax] || xMean, xRange / 6, yMin],
    ["a", "x₀", "γ", "c"], [2]);

  // Sine (multi-start with frequency estimation)
  const estOmega = estimateFrequency();
  const sineSeeds = [[yRange / 2, estOmega, 0, yMean], [yRange / 2, 2 * Math.PI / xRange, 0, yMean], [yRange / 2, estOmega * 2, Math.PI / 4, yMean]];
  sineSeeds.forEach((seed, si) => add("Sine", si === 0 ? "Sine Wave" : `Sine (seed ${si + 1})`,
    "y = a·sin(ω·x + φ) + d", 4,
    (x, p) => p[0] * Math.sin(p[1] * x + p[2]) + p[3],
    seed, ["a", "ω", "φ", "d"], [0, 1]));

  return models;
}

// ============================================================
// DATA PARSING
// ============================================================
function parseData(text) {
  const lines = text.trim().split(/\r?\n/).filter(l => l.trim());
  if (lines.length < 2) return null;
  const dataLine = lines[Math.min(1, lines.length - 1)];

  // Detect separator: tab > semicolon > comma > whitespace
  let sep;
  if (dataLine.includes('\t')) {
    sep = '\t';
  } else if ((dataLine.split(';').length - 1) > (dataLine.split(',').length - 1)) {
    sep = ';';
  } else if (dataLine.includes(',')) {
    sep = ',';
  } else if (/\s{2,}/.test(dataLine) || /^\s*\S+\s+\S+/.test(dataLine)) {
    sep = 'SPACE';
  } else {
    sep = ',';
  }

  // European decimal comma detection: semicolon separator + commas inside numbers
  // e.g. "1,5;10,3" → sep=';', values contain commas as decimal separators
  let euDecimal = false;
  if (sep === ';') {
    const testParts = dataLine.split(';').map(s => s.trim());
    const hasCommaNumbers = testParts.some(p => /^\d+,\d+$/.test(p));
    const allParseOk = testParts.every(p => isFinite(parseFloat(p)));
    if (hasCommaNumbers && !allParseOk) {
      euDecimal = true;
    }
  }

  const splitLine = (line) => {
    if (sep === 'SPACE') return line.trim().split(/\s+/);
    return line.split(sep);
  };

  const parseNum = (s) => {
    let v = s.trim();
    if (euDecimal) v = v.replace(',', '.');
    return parseFloat(v);
  };

  const firstParts = splitLine(lines[0]).map(s => s.trim());
  const hasHeader = firstParts.some(p => isNaN(parseNum(p)) && p.length > 0);
  const startIdx = hasHeader ? 1 : 0;
  const headers = hasHeader ? firstParts : ['x', 'y'];
  const xData = [], yData = [];
  for (let i = startIdx; i < lines.length; i++) {
    const parts = splitLine(lines[i]).map(s => parseNum(s));
    if (parts.length >= 2 && isFinite(parts[0]) && isFinite(parts[1])) { xData.push(parts[0]); yData.push(parts[1]); }
  }
  return xData.length >= 3 ? { xData, yData, headers: [headers[0] || 'x', headers[1] || 'y'], n: xData.length } : null;
}

// ============================================================
// CUSTOM MODEL PARSER — SECURE (no eval / no new Function)
// Recursive-descent parser: tokenize → parse AST → compile to closure
// Supports: +, -, *, /, ^, unary -, (), exp, log, ln, sqrt, abs, sin, cos, tan, pow, PI
// Parameters: single letters (a-z except x,e) or subscripted (a1, k2)
// Security: rejects ANY character outside strict whitelist (no [], {}, ;, =, etc.)
// ============================================================
function parseCustomModel(text) {
  const trimmed = text.trim();
  if (!trimmed) return null;

  let name = "Custom Model", expr = trimmed;
  const colonIdx = trimmed.indexOf(':');
  if (colonIdx > 0 && colonIdx < 30 && !trimmed.substring(0, colonIdx).includes('(')) {
    name = trimmed.substring(0, colonIdx).trim();
    expr = trimmed.substring(colonIdx + 1).trim();
  }
  expr = expr.replace(/^y\s*=\s*/i, '');
  if (!expr) return null;

  // ---- TOKENIZER (strict whitelist) ----
  const FUNCS = new Set(['exp', 'log', 'ln', 'sqrt', 'abs', 'sin', 'cos', 'tan', 'pow']);
  const reserved = new Set(['x', 'e', 'exp', 'log', 'ln', 'sin', 'cos', 'tan', 'sqrt', 'abs', 'pow', 'min', 'max', 'pi']);
  const tokens = [];
  const paramSet = new Set();
  let ti = 0;

  while (ti < expr.length) {
    const ch = expr[ti];
    if (/\s/.test(ch)) { ti++; continue; }

    // Number (including decimals and scientific notation like 1.5e-3)
    if (/[0-9.]/.test(ch)) {
      let num = '';
      while (ti < expr.length && /[0-9.]/.test(expr[ti])) num += expr[ti++];
      if (ti < expr.length && /[eE]/.test(expr[ti])) {
        num += expr[ti++];
        if (ti < expr.length && /[+\-]/.test(expr[ti])) num += expr[ti++];
        while (ti < expr.length && /[0-9]/.test(expr[ti])) num += expr[ti++];
      }
      const val = parseFloat(num);
      if (!isFinite(val)) return null;
      tokens.push({ type: 'NUM', value: val });
      continue;
    }

    // Word (function, variable, parameter, constant)
    if (/[a-zA-Z]/.test(ch)) {
      let word = '';
      while (ti < expr.length && /[a-zA-Z0-9]/.test(expr[ti])) word += expr[ti++];
      const lw = word.toLowerCase();
      if (lw === 'pi') { tokens.push({ type: 'NUM', value: Math.PI }); }
      else if (lw === 'x') { tokens.push({ type: 'VAR' }); }
      else if (FUNCS.has(lw)) { tokens.push({ type: 'FUNC', value: lw }); }
      else if (/^[a-z]\d*$/.test(lw) && lw.length <= 3 && !reserved.has(lw)) {
        paramSet.add(lw); tokens.push({ type: 'PARAM', value: lw });
      }
      else return null; // Unknown identifier — reject
      continue;
    }

    // Operators and parens (strict whitelist)
    if ('+-*/^(),'.includes(ch)) { tokens.push({ type: ch }); ti++; continue; }
    // Unicode minus (−) and multiplication (·, ×)
    if (ch === '\u2212' || ch === '\u2013') { tokens.push({ type: '-' }); ti++; continue; }
    if (ch === '\u00B7' || ch === '\u00D7') { tokens.push({ type: '*' }); ti++; continue; }

    return null; // REJECT any unexpected character (security: no [], {}, ;, =, ?, :, ', ", `, \, <, > etc.)
  }

  const paramNames = Array.from(paramSet).sort();
  if (paramNames.length === 0 || paramNames.length > 8) return null;
  const paramIdx = Object.fromEntries(paramNames.map((n, i) => [n, i]));

  // ---- RECURSIVE DESCENT PARSER → AST ----
  let pos = 0;
  const peek = () => pos < tokens.length ? tokens[pos] : null;
  const consume = (type) => { const t = peek(); if (t && t.type === type) { pos++; return t; } return null; };
  const expect = (type) => { const t = consume(type); if (!t) throw new Error('parse'); return t; };

  // Grammar:
  //   expr  → term (('+' | '-') term)*
  //   term  → unary (('*' | '/') unary)*   [with implicit multiply detection]
  //   unary → '-' unary | '+' unary | power
  //   power → atom ('^' unary)?
  //   atom  → NUM | VAR | PARAM | FUNC '(' expr (',' expr)? ')' | '(' expr ')'

  function parseExpr() {
    let node = parseTerm();
    while (peek() && (peek().type === '+' || peek().type === '-')) {
      const op = tokens[pos++].type;
      node = { op, left: node, right: parseTerm() };
    }
    return node;
  }

  function parseTerm() {
    let node = parseUnary();
    while (true) {
      const next = peek();
      if (!next) break;
      // Explicit multiply/divide
      if (next.type === '*' || next.type === '/') {
        const op = tokens[pos++].type;
        node = { op, left: node, right: parseUnary() };
      }
      // Implicit multiplication: 2x, 2a, a(x+1), 2(3+x), etc.
      else if (next.type === 'NUM' || next.type === 'VAR' || next.type === 'PARAM' || next.type === 'FUNC' || next.type === '(') {
        node = { op: '*', left: node, right: parseUnary() };
      }
      else break;
    }
    return node;
  }

  function parseUnary() {
    if (peek() && peek().type === '-') { pos++; return { op: 'neg', child: parseUnary() }; }
    if (peek() && peek().type === '+') { pos++; return parseUnary(); }
    return parsePower();
  }

  function parsePower() {
    let node = parseAtom();
    if (peek() && peek().type === '^') { pos++; node = { op: '^', left: node, right: parseUnary() }; }
    return node;
  }

  function parseAtom() {
    const t = peek();
    if (!t) throw new Error('parse');
    if (t.type === 'NUM') { pos++; return { op: 'num', value: t.value }; }
    if (t.type === 'VAR') { pos++; return { op: 'var' }; }
    if (t.type === 'PARAM') { pos++; return { op: 'param', index: paramIdx[t.value] }; }
    if (t.type === 'FUNC') {
      pos++;
      expect('(');
      const arg1 = parseExpr();
      let arg2 = null;
      if (peek() && peek().type === ',') { pos++; arg2 = parseExpr(); }
      expect(')');
      return { op: 'func', name: t.value, arg1, arg2 };
    }
    if (t.type === '(') { pos++; const node = parseExpr(); expect(')'); return node; }
    throw new Error('parse');
  }

  let ast;
  try {
    ast = parseExpr();
    if (pos !== tokens.length) return null; // Trailing tokens — reject
  } catch { return null; }

  // ---- COMPILE AST → CLOSURE (no eval, no Function constructor) ----
  function compile(node) {
    switch (node.op) {
      case 'num': { const v = node.value; return () => v; }
      case 'var': return (x) => x;
      case 'param': { const idx = node.index; return (x, p) => p[idx]; }
      case 'neg': { const c = compile(node.child); return (x, p, E) => -c(x, p, E); }
      case '+': { const l = compile(node.left), r = compile(node.right); return (x, p, E) => l(x, p, E) + r(x, p, E); }
      case '-': { const l = compile(node.left), r = compile(node.right); return (x, p, E) => l(x, p, E) - r(x, p, E); }
      case '*': { const l = compile(node.left), r = compile(node.right); return (x, p, E) => l(x, p, E) * r(x, p, E); }
      case '/': { const l = compile(node.left), r = compile(node.right); return (x, p, E) => { const d = r(x, p, E); return d === 0 ? NaN : l(x, p, E) / d; }; }
      case '^': { const l = compile(node.left), r = compile(node.right); return (x, p, E) => safePow(l(x, p, E), r(x, p, E)); }
      case 'func': {
        const a1 = compile(node.arg1);
        const a2 = node.arg2 ? compile(node.arg2) : null;
        switch (node.name) {
          case 'exp': return (x, p, E) => (E || safeExp)(a1(x, p, E));
          case 'log': case 'ln': return (x, p, E) => Math.log(a1(x, p, E));
          case 'sqrt': return (x, p, E) => Math.sqrt(a1(x, p, E));
          case 'abs': return (x, p, E) => Math.abs(a1(x, p, E));
          case 'sin': return (x, p, E) => Math.sin(a1(x, p, E));
          case 'cos': return (x, p, E) => Math.cos(a1(x, p, E));
          case 'tan': return (x, p, E) => Math.tan(a1(x, p, E));
          case 'pow': return a2 ? (x, p, E) => safePow(a1(x, p, E), a2(x, p, E)) : () => NaN;
          default: return () => NaN;
        }
      }
      default: return () => NaN;
    }
  }

  const fn = compile(ast);

  // Validate: test evaluation
  try {
    const testVal = fn(1, new Array(paramNames.length).fill(1), safeExp);
    if (typeof testVal !== 'number') return null;
  } catch { return null; }

  // Auto-detect positive parameters: params inside exp() subtrees containing negation
  // Covers: exp(-b*x), exp(-k*x+c), a*exp(neg(...)), etc.
  const positiveIdx = [];
  function hasNegation(node) {
    if (!node) return false;
    if (node.op === 'neg') return true;
    return hasNegation(node.left) || hasNegation(node.right) || hasNegation(node.child)
      || hasNegation(node.arg1) || hasNegation(node.arg2);
  }
  function collectParams(node) {
    if (!node) return;
    if (node.op === 'param' && !positiveIdx.includes(node.index)) positiveIdx.push(node.index);
    collectParams(node.left); collectParams(node.right); collectParams(node.child);
    collectParams(node.arg1); collectParams(node.arg2);
  }
  (function findExpPatterns(node) {
    if (!node) return;
    if (node.op === 'func' && node.name === 'exp' && hasNegation(node.arg1)) {
      // exp contains negation → all params inside are likely rate/decay constants
      collectParams(node.arg1);
    }
    findExpPatterns(node.left); findExpPatterns(node.right); findExpPatterns(node.child);
    findExpPatterns(node.arg1); findExpPatterns(node.arg2);
  })(ast);

  const init = new Array(paramNames.length).fill(1);
  return {
    family: 'Custom', name, equation: `y = ${expr}`, nParams: paramNames.length,
    func: (x, p, E) => { try { return fn(x, p, E); } catch { return NaN; } },
    init, paramNames, positiveIdx
  };
}

// ============================================================
// SAMPLE DATASETS
// ============================================================
const SAMPLES = {
  "Enzyme Kinetics": "Substrate_Concentration,Reaction_Rate\n0.1,1.8\n0.2,3.2\n0.5,6.1\n1.0,8.9\n2.0,11.5\n5.0,14.2\n10.0,15.8\n20.0,16.9\n50.0,17.5\n100.0,17.8",
  "Dose-Response": "Concentration_nM,Response_%\n0.01,2.1\n0.03,3.8\n0.1,8.5\n0.3,18.2\n1,42.1\n3,68.5\n10,85.3\n30,94.1\n100,97.8\n300,99.2",
  "Bacterial Growth": "Time_hours,Population_OD600\n0,0.05\n1,0.08\n2,0.15\n3,0.32\n4,0.65\n5,1.15\n6,1.72\n7,2.10\n8,2.35\n9,2.48\n10,2.55\n11,2.58\n12,2.59",
  "Radioactive Decay": "Time_min,Activity_Bq\n0,1000\n5,820\n10,672\n15,551\n20,452\n25,370\n30,304\n40,204\n50,137\n60,92\n80,41\n100,18",
  "Gaussian Peak": "Wavelength_nm,Intensity\n400,2.1\n420,3.5\n440,8.2\n450,15.3\n455,22.1\n460,30.5\n465,35.2\n470,32.8\n475,25.1\n480,16.4\n490,7.8\n500,3.9\n520,2.3",
  "Adsorption Isotherm": "Pressure_kPa,Loading_mmol_g\n0.5,0.82\n1.0,1.45\n2.0,2.31\n5.0,3.78\n10.0,4.89\n20.0,5.72\n50.0,6.45\n100.0,6.82\n200.0,7.05\n500.0,7.18"
};

// ============================================================
// MAIN COMPONENT v4
// ============================================================
export default function CurveFitter() {
  const [rawText, setRawText] = useState("");
  const [data, setData] = useState(null);
  const [results, setResults] = useState(null);
  const [selectedModel, setSelectedModel] = useState(null);
  const [fitting, setFitting] = useState(false);
  const [fitProgress, setFitProgress] = useState(0);
  const [error, setError] = useState(null);
  const [showResiduals, setShowResiduals] = useState(false);
  const [showCI, setShowCI] = useState(true);
  const [logX, setLogX] = useState(false);
  const [logY, setLogY] = useState(false);
  const [customExpr, setCustomExpr] = useState("");
  const [customError, setCustomError] = useState(null);
  const [proModalOpen, setProModalOpen] = useState(false);
  const [copied, setCopied] = useState(null);
  const [isDemo, setIsDemo] = useState(true);
  const chartRef = useRef(null);
  const fitGenRef = useRef(0); // generation counter to cancel stale fits
  const demoInitRef = useRef(false);
  const handleFitRef = useRef(null);

  const handleParse = useCallback((text, { demo = false } = {}) => {
    fitGenRef.current++; // cancel any in-flight fit
    setRawText(text); setError(null); setResults(null); setSelectedModel(null); setFitting(false);
    if (!demo) setIsDemo(false);
    const parsed = parseData(text);
    if (parsed) setData(parsed);
    else if (text.trim().length > 0) { setError("Need ≥3 rows with 2 numeric columns."); setData(null); }
  }, []);

  const handleFile = (e) => {
    const file = e.target.files[0]; if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => handleParse(ev.target.result);
    reader.readAsText(file);
  };

  // Auto-load demo data on mount
  useEffect(() => {
    if (demoInitRef.current) return;
    demoInitRef.current = true;
    handleParse(SAMPLES["Enzyme Kinetics"], { demo: true });
  }, [handleParse]);

  // Auto-fit once demo data is parsed
  useEffect(() => {
    if (isDemo && data && !results && !fitting) {
      handleFitRef.current?.();
    }
  }, [isDemo, data, results, fitting]);

  const handleFit = useCallback(() => {
    if (!data) return;
    const gen = ++fitGenRef.current; // capture generation for this fit
    setFitting(true); setFitProgress(0); setError(null); setCustomError(null);
    const models = buildModels(data.xData, data.yData);

    // Inject custom model if provided
    if (customExpr.trim()) {
      const custom = parseCustomModel(customExpr);
      if (custom) {
        // Smart initial guesses from data
        const yRange = Math.max(...data.yData) - Math.min(...data.yData);
        custom.init = custom.init.map(() => yRange > 0 ? yRange * 0.5 : 1);
        models.push(custom);
      } else {
        setCustomError("Could not parse expression. Use: a * exp(-b * x) + c");
      }
    }

    const total = models.length;
    const fits = [];
    let idx = 0;

    const fitNext = () => {
      // Abort if data changed since fit started
      if (gen !== fitGenRef.current) { setFitting(false); return; }
      const end = Math.min(idx + 3, total);
      for (let i = idx; i < end; i++) {
        const m = models[i];
        try {
          const r = levenbergMarquardt(m.func, data.xData, data.yData, m.init, { positiveIdx: m.positiveIdx || [] });
          // CRITICAL: Disqualify models whose final params produce non-finite predictions
          // This catches cases like Hill with n=251563 where x^n overflows to Infinity
          if (isFinite(r.aicc) && r.finalInvalidCount === 0) fits.push({ ...m, ...r });
        } catch {}
      }
      idx = end;
      setFitProgress(Math.round((idx / total) * 100));

      if (idx < total) {
        setTimeout(fitNext, 0);
      } else {
        // FIX: Dedup by family field
        const bestByFamily = new Map();
        for (const fit of fits) {
          const existing = bestByFamily.get(fit.family);
          if (!existing || fit.aicc < existing.aicc) {
            bestByFamily.set(fit.family, fit);
          }
        }
        const deduped = Array.from(bestByFamily.values());
        deduped.sort((a, b) => a.aicc - b.aicc);
        const bestAicc = deduped.length > 0 ? deduped[0].aicc : 0;

        // Akaike weights
        const rawWeights = deduped.map(f => Math.exp(-0.5 * (f.aicc - bestAicc)));
        const wSum = rawWeights.reduce((s, w) => s + w, 0);
        deduped.forEach((f, i) => {
          f.deltaAicc = f.aicc - bestAicc;
          f.akaikeWeight = wSum > 0 ? rawWeights[i] / wSum : 0;

          // Quality score: ✅ Good / ⚠ Needs review / ❌ Unreliable
          const warnCount = (f.warnings || []).length;
          const hasNegR2 = f.rSquared < 0;
          const lowDof = f.dof < 3;
          const noCi = !f.stdErrors;
          const poorFit = f.adjR2 < 0.5;
          const hasClampWarn = (f.warnings || []).some(w => w.includes('clamp'));
          const hasCorrelWarn = (f.warnings || []).some(w => w.includes('correlation'));

          if (hasNegR2 || lowDof || (warnCount >= 3) || (poorFit && hasClampWarn)) {
            f.quality = { label: "Unreliable", color: "text-red-400", bg: "bg-red-500/15", icon: "✗" };
          } else if (warnCount >= 1 || poorFit || noCi || hasCorrelWarn) {
            f.quality = { label: "Needs review", color: "text-yellow-400", bg: "bg-yellow-500/15", icon: "?" };
          } else {
            f.quality = { label: "Good", color: "text-green-400", bg: "bg-green-500/15", icon: "✓" };
          }
        });

        setResults(deduped);
        setSelectedModel(deduped.length > 0 ? 0 : null);
        setFitting(false);
      }
    };
    setTimeout(fitNext, 10);
  }, [data, customExpr]);

  // Keep ref in sync so useEffect can call without circular deps
  handleFitRef.current = handleFit;

  const chartDataMemo = useMemo(() => {
    if (!data) return { dp: [], fp: [], xDomain: undefined };
    const dp = data.xData.map((x, i) => ({ x, y: data.yData[i] }))
      .filter(p => (!logX || p.x > 0) && (!logY || p.y > 0));
    const fp = [];
    let xDomain;
    if (results && selectedModel !== null && results[selectedModel]) {
      const m = results[selectedModel];
      const hasBands = m.covMatrix && m.s2 && m.tVal;
      const validX = logX ? data.xData.filter(x => x > 0) : data.xData;
      const xMin = Math.min(...validX), xMax = Math.max(...validX);

      const computePoint = (x) => {
        try {
          const y = m.func(x, m.params, safeExp);
          if (!isFinite(y) || (logY && y <= 0)) return null;

          const pt = { x, yFit: y };

          // Confidence band: σ²_y = s² · gᵀ · (JᵀJ)⁻¹ · g
          if (hasBands && showCI) {
            const p = m.nParams || m.params.length;
            const g = new Array(p);
            for (let j = 0; j < p; j++) {
              const epsJ = 1e-6 * (Math.abs(m.params[j]) + 1);
              const pp = [...m.params]; pp[j] += epsJ;
              const pm = [...m.params]; pm[j] -= epsJ;
              const f1 = m.func(x, pp, safeExp);
              const f2 = m.func(x, pm, safeExp);
              g[j] = (isFinite(f1) && isFinite(f2)) ? (f1 - f2) / (2 * epsJ) : 0;
            }
            // gᵀ · Cov · g  (Cov already is (JᵀJ)⁻¹, multiply by s²)
            let varY = 0;
            for (let j = 0; j < p; j++) {
              for (let k = 0; k < p; k++) {
                varY += g[j] * m.covMatrix[j][k] * g[k];
              }
            }
            varY *= m.s2;
            if (varY > 0 && isFinite(varY)) {
              const band = m.tVal * Math.sqrt(varY);
              pt.bandUpper = y + band;
              pt.bandLower = y - band;
            }
          }
          return pt;
        } catch { return null; }
      };

      // Fewer points when computing CI bands (2p gradient evals per point)
      const nPts = (hasBands && showCI) ? 150 : 300;

      const margin = 0.05;
      if (logX) {
        const logMin = Math.log10(xMin), logMax = Math.log10(xMax);
        const logRange = logMax - logMin;
        for (let i = 0; i <= nPts; i++) {
          const x = Math.pow(10, logMin - logRange * margin + (logRange * (1 + 2 * margin)) * i / nPts);
          const pt = computePoint(x);
          if (pt) fp.push(pt);
        }
      } else {
        const range = xMax - xMin || 1;
        const lo = xMin - range * margin;
        const hi = xMax + range * margin;
        for (let i = 0; i <= nPts; i++) {
          const x = lo + (hi - lo) * i / nPts;
          const pt = computePoint(x);
          if (pt) fp.push(pt);
        }
        xDomain = [lo, hi];
      }

      // Clamp fitted curve & bands so extreme model values don't blow up the Y axis
      if (fp.length > 0 && dp.length > 0) {
        const yVals = dp.map(p => p.y);
        const dataYMin = Math.min(...yVals);
        const dataYMax = Math.max(...yVals);
        const yRange = (dataYMax - dataYMin) || 1;
        const clampHi = dataYMax + 3 * yRange;
        const clampLo = dataYMin - 3 * yRange;
        for (let i = fp.length - 1; i >= 0; i--) {
          const pt = fp[i];
          if (!isFinite(pt.yFit)) { fp.splice(i, 1); continue; }
          pt.yFit = Math.max(clampLo, Math.min(clampHi, pt.yFit));
          if (pt.bandUpper != null) {
            pt.bandUpper = Math.max(clampLo, Math.min(clampHi, pt.bandUpper));
            pt.bandLower = Math.max(clampLo, Math.min(clampHi, pt.bandLower));
          }
        }
      }
    }
    return { dp, fp, xDomain };
  }, [data, results, selectedModel, logX, logY, showCI]);

  const residualData = useMemo(() => {
    if (!data || !results || selectedModel === null) return [];
    const m = results[selectedModel];
    return data.xData.map((x, i) => {
      const val = m.func(x, m.params, safeExp);
      const r = isFinite(val) ? data.yData[i] - val : null;
      return { x, residual: r, zero: 0 };
    }).filter(d => d.residual !== null);
  }, [data, results, selectedModel]);

  const fmt = (v) => { if (!isFinite(v)) return "—"; if (Math.abs(v) < 0.001 || Math.abs(v) > 99999) return v.toExponential(4); return v.toPrecision(6); };

  // Nice tick computation for clean axis labels (linear mode only)
  const niceTicks = (dMin, dMax, count = 5) => {
    if (!isFinite(dMin) || !isFinite(dMax) || dMin === dMax) {
      return dMin === dMax && isFinite(dMin) ? [dMin] : [];
    }
    const range = dMax - dMin;
    const rawStep = range / count;
    const mag = Math.pow(10, Math.floor(Math.log10(rawStep)));
    const residual = rawStep / mag;
    let niceStep;
    if (residual < 1.5) niceStep = 1 * mag;
    else if (residual < 3) niceStep = 2 * mag;
    else if (residual < 7) niceStep = 5 * mag;
    else niceStep = 10 * mag;
    const iLo = Math.floor(dMin / niceStep);
    const iHi = Math.ceil(dMax / niceStep);
    const ticks = [];
    for (let i = iLo; i <= iHi; i++) {
      ticks.push(parseFloat((i * niceStep).toPrecision(12)));
    }
    return ticks;
  };

  const linearTickFmt = (v) => parseFloat(v.toPrecision(6));

  // Precompute nice ticks for linear axes
  const axisTicks = useMemo(() => {
    if (!data) return { x: undefined, y: undefined };
    const xs = data.xData, ys = data.yData;
    const xMin = Math.min(...xs), xMax = Math.max(...xs);
    const yMin = Math.min(...ys), yMax = Math.max(...ys);
    return {
      x: logX ? undefined : niceTicks(xMin, xMax, 5),
      y: logY ? undefined : niceTicks(yMin, yMax, 5),
    };
  }, [data, logX, logY]);

  const copyParams = (model) => {
    const lines = model.paramNames.map((n, i) => {
      const se = model.stdErrors && isFinite(model.stdErrors[i]) ? ` ± ${fmt(model.stdErrors[i])}` : '';
      return `${n} = ${fmt(model.params[i])}${se}`;
    });
    lines.push(`R² = ${model.rSquared.toFixed(6)}`);
    lines.push(`adj. R² = ${model.adjR2.toFixed(6)}`);
    lines.push(`AICc = ${model.aicc.toFixed(2)}`);
    const text = `${model.name}: ${model.equation}\n${lines.join('\n')}`;
    navigator.clipboard.writeText(text).then(() => {
      setCopied(model.name);
      setTimeout(() => setCopied(null), 2000);
    }).catch(() => {});
  };

  // Model equivalence notes
  const MODEL_NOTES = {
    'Freundlich': 'Mathematically equivalent to Power Law (y = Kf·x^(1/n) ≡ a·x^b). Both are shown for scientific naming conventions.',
    'PowerLaw': 'Mathematically equivalent to Freundlich Isotherm. Compare carefully if both rank highly.',
    '5PL': 'Asymmetric extension of 4PL — the g parameter controls asymmetry. When g=1, reduces to 4PL.',
    'BiExp': 'Two decay components (fast + slow). Parameters (a₁,k₁) and (a₂,k₂) are interchangeable — CIs may be inflated due to this symmetry.',
    'Gompertz': 'Asymmetric sigmoid — unlike Logistic, the inflection point is not at the midpoint.',
    'Lorentzian': 'Cauchy peak profile — heavier tails than Gaussian. Common in spectroscopy (NMR, XRD).',
    'Custom': 'User-defined equation. Rate params in exp(-k·x) auto-detected as positive. Note: a^b^c = a^(b^c).',
  };

  const handleExportSVG = () => {
    const svgEl = chartRef.current?.querySelector('svg'); if (!svgEl) return;
    const clone = svgEl.cloneNode(true);
    clone.querySelectorAll('*').forEach(el => {
      const c = window.getComputedStyle(el);
      ['fill', 'stroke', 'strokeWidth', 'fontSize', 'fontFamily'].forEach(p => { if (c[p]) el.setAttribute(p.replace(/([A-Z])/g, '-$1').toLowerCase(), c[p]); });
    });
    const blob = new Blob([new XMLSerializer().serializeToString(clone)], { type: 'image/svg+xml' });
    const a = document.createElement('a'); const url = URL.createObjectURL(blob); a.href = url; a.download = 'curvefit.svg'; a.click(); URL.revokeObjectURL(url);
  };

  const handleExportCSV = () => {
    if (!results) return;
    const lines = ["Model,Family,Quality,AICc,ΔAICc,Akaike_Weight,Adj_R²,R²,DOF,Parameters,Std_Errors,Warnings"];
    for (const r of results) {
      const params = r.paramNames.map((n, i) => `${n}=${fmt(r.params[i])}`).join('; ');
      const se = r.stdErrors ? r.paramNames.map((n, i) => `${n}=±${fmt(r.stdErrors[i])}`).join('; ') : 'N/A';
      const w = (r.warnings || []).join(' | ');
      const q = r.quality ? r.quality.label : '';
      lines.push(`"${r.name}","${r.family}","${q}",${r.aicc.toFixed(2)},${r.deltaAicc.toFixed(2)},${r.akaikeWeight.toFixed(4)},${r.adjR2.toFixed(6)},${r.rSquared.toFixed(6)},${r.dof},"${params}","${se}","${w}"`);
    }
    const blob = new Blob([lines.join('\n')], { type: 'text/csv' });
    const a = document.createElement('a'); const url = URL.createObjectURL(blob); a.href = url; a.download = 'curvefit_results.csv'; a.click(); URL.revokeObjectURL(url);
  };

  const sel = results && selectedModel !== null ? results[selectedModel] : null;

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 p-3" style={{ fontFamily: "'Inter', system-ui, sans-serif" }}>
      <div className="max-w-6xl mx-auto">
        <header className="mb-5 flex items-start justify-between">
          <div>
            <h1 className="text-3xl font-bold text-white tracking-tight">
              <span className="text-blue-400">Curve</span>Fit
            </h1>
            <p className="text-gray-400 mt-1 text-xs">
              25+ scientific models · Custom equations · AICc ranking · Akaike weights · Confidence intervals · Publication-ready · Client-side only
            </p>
          </div>
          <button onClick={() => setProModalOpen(true)}
            className="flex items-center gap-1.5 text-xs px-3 py-1.5 bg-gray-800 hover:bg-gray-700 rounded text-gray-300 border border-gray-700 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 shrink-0">
            <span className="text-amber-400">🔒</span> Save Project <span className="text-[10px] font-semibold bg-amber-500/20 text-amber-400 px-1.5 py-0.5 rounded-full leading-none">PRO</span>
          </button>
        </header>

        {isDemo && results && (
          <div className="mb-3 flex items-center justify-between rounded-lg bg-blue-600/15 border border-blue-500/30 px-4 py-2.5">
            <p className="text-sm text-blue-200">
              Showing demo with enzyme kinetics data — paste your own data to get started.
            </p>
            <button onClick={() => setIsDemo(false)} className="text-blue-300 hover:text-white text-xs ml-4 shrink-0 focus:outline-none">✕ Dismiss</button>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
          {/* LEFT — Input & Ranking */}
          <nav className="lg:col-span-1 space-y-3" aria-label="Data input and model ranking">
            <section className="bg-gray-900 rounded-lg p-3 border border-gray-800">
              <h2 className="text-xs font-medium text-gray-400 uppercase tracking-wide">Sample data</h2>
              <div className="flex flex-wrap gap-1.5 mt-2" role="group" aria-label="Sample datasets">
                {Object.keys(SAMPLES).map(name => (
                  <button key={name} onClick={() => handleParse(SAMPLES[name])}
                    className="text-xs px-2 py-1 bg-gray-800 hover:bg-gray-700 rounded text-gray-300 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500">{name}</button>
                ))}
              </div>
            </section>

            <section className="bg-gray-900 rounded-lg p-3 border border-gray-800">
              <div className="flex items-center justify-between mb-2">
                <h2 className="text-xs font-medium text-gray-400 uppercase tracking-wide">Your data</h2>
                <label className="text-xs px-2 py-1 bg-blue-600 hover:bg-blue-500 rounded cursor-pointer transition-colors focus-within:ring-2 focus-within:ring-blue-400">
                  Upload <input type="file" accept=".csv,.tsv,.txt" onChange={handleFile} className="hidden" aria-label="Upload CSV, TSV, or TXT file" />
                </label>
              </div>
              <textarea className="w-full h-36 bg-gray-800 rounded p-2 text-xs font-mono text-gray-300 border border-gray-700 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 resize-none"
                placeholder={"x, y\n0.1, 1.8\n0.2, 3.2\n..."} value={rawText} onChange={(e) => handleParse(e.target.value)}
                aria-label="Paste your data here" />
              {data && <p className="text-xs text-green-400 mt-1" role="status">✓ {data.n} points parsed</p>}
              {error && <p className="text-xs text-red-400 mt-1" role="alert">✗ {error}</p>}
            </section>

            {/* Custom Model */}
            <section className="bg-gray-900 rounded-lg p-3 border border-gray-800">
              <h2 className="text-xs font-medium text-gray-400 uppercase tracking-wide mb-2">Custom model <span className="text-gray-400 normal-case font-normal">(optional)</span></h2>
              <input type="text" value={customExpr} onChange={(e) => { setCustomExpr(e.target.value); setCustomError(null); }}
                className="w-full bg-gray-800 rounded p-2 text-xs font-mono text-gray-300 border border-gray-700 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
                placeholder="a * exp(-b * x) + c"
                aria-label="Custom model equation" />
              <p className="text-xs text-gray-400 mt-1">Use x as variable, a–z as parameters. Functions: exp, log, sin, cos, sqrt, pow. Rate constants in exp(−k·x) are auto-constrained positive.</p>
              {customError && <p className="text-xs text-red-400 mt-1" role="alert">✗ {customError}</p>}
            </section>

            <button onClick={handleFit} disabled={!data || fitting}
              className={`w-full py-3 rounded-lg font-semibold text-sm transition-all focus:outline-none focus:ring-2 focus:ring-blue-400 ${data && !fitting ? 'bg-blue-600 hover:bg-blue-500 text-white shadow-lg shadow-blue-600/20' : 'bg-gray-800 text-gray-500 cursor-not-allowed'}`}
              aria-busy={fitting}>
              {fitting ? `Fitting... ${fitProgress}%` : "⚡ Auto-Fit All Models"}
            </button>
            {fitting && (
              <div className="w-full bg-gray-800 rounded-full h-1.5" role="progressbar" aria-valuenow={fitProgress} aria-valuemin={0} aria-valuemax={100}>
                <div className="bg-blue-500 h-1.5 rounded-full transition-all" style={{ width: `${fitProgress}%` }} />
              </div>
            )}

            {results && results.length === 0 && !fitting && (
              <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3" role="alert">
                <p className="text-red-400 text-xs font-medium">No models converged successfully.</p>
                <p className="text-red-400/70 text-xs mt-1">
                  {data && data.n <= 4
                    ? `Dataset has only ${data.n} points — most models need ≥5 points for reliable fitting. Try adding more data.`
                    : "All models produced non-finite predictions or failed to converge. Check your data for extreme values or formatting issues."}
                </p>
              </div>
            )}

            {results && results.length > 0 && (
              <section className="bg-gray-900 rounded-lg border border-gray-800 overflow-hidden">
                <div className="p-2 border-b border-gray-800 flex justify-between items-center">
                  <h2 className="text-xs font-medium text-gray-400 uppercase tracking-wide">Ranking (AICc)</h2>
                  <span className="text-xs text-gray-500">{results.length} models</span>
                </div>
                <div className="max-h-72 overflow-y-auto" role="listbox" aria-label="Model ranking" tabIndex={0}
                  onKeyDown={(e) => {
                    if (e.key === 'ArrowDown' && selectedModel < results.length - 1) { e.preventDefault(); setSelectedModel(selectedModel + 1); }
                    if (e.key === 'ArrowUp' && selectedModel > 0) { e.preventDefault(); setSelectedModel(selectedModel - 1); }
                  }}>
                  {results.map((r, i) => (
                    <button key={i} onClick={() => setSelectedModel(i)}
                      role="option" aria-selected={selectedModel === i}
                      className={`w-full text-left px-3 py-2 text-xs border-b border-gray-800/50 transition-colors focus:outline-none focus:ring-1 focus:ring-inset focus:ring-blue-500 ${selectedModel === i ? 'bg-blue-600/20 text-blue-200' : 'hover:bg-gray-800 text-gray-300'}`}>
                      <div className="flex justify-between items-center gap-1">
                        <div className="flex items-center gap-1.5 truncate">
                          {i === 0 && <span className="bg-blue-500 text-white text-xs font-bold px-1.5 py-0.5 rounded" aria-label="Best model">BEST</span>}
                          {r.quality && <span className={`${r.quality.color} font-bold text-xs`} aria-label={r.quality.label}>{r.quality.icon}</span>}
                          <span className="font-medium truncate">{r.name}</span>
                          {r.family === 'Custom' && <span className="text-blue-400 text-xs">★</span>}
                        </div>
                        <div className="flex gap-2 shrink-0 items-center">
                          <span className={`font-mono text-xs ${r.adjR2 > 0.99 ? 'text-green-400' : r.adjR2 > 0.95 ? 'text-yellow-400' : r.adjR2 < 0 ? 'text-red-500' : 'text-gray-400'}`}>
                            {r.adjR2 < 0 ? `R²=${r.adjR2.toFixed(2)}` : `R²=${r.adjR2.toFixed(4)}`}
                          </span>
                          <span className="font-mono text-gray-400 text-xs">{(r.akaikeWeight * 100).toFixed(0)}%</span>
                        </div>
                      </div>
                    </button>
                  ))}
                </div>
              </section>
            )}
          </nav>

          {/* RIGHT — Chart & Details */}
          <main className="lg:col-span-2 space-y-3" aria-label="Results">
            <section className="bg-gray-900 rounded-lg p-4 border border-gray-800" ref={chartRef} aria-label="Curve fit chart">
              {data && (
                <div className="flex gap-2 mb-3" role="toolbar" aria-label="Chart controls">
                  <button onClick={() => setLogX(!logX)} aria-pressed={logX}
                    className={`text-xs px-3 py-1 rounded transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 ${logX ? 'bg-blue-600 text-white' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'}`}>
                    Log X
                  </button>
                  <button onClick={() => setLogY(!logY)} aria-pressed={logY}
                    className={`text-xs px-3 py-1 rounded transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 ${logY ? 'bg-blue-600 text-white' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'}`}>
                    Log Y
                  </button>
                  {(logX || logY) && <span className="text-xs text-gray-500 self-center">Points ≤0 hidden on log axes</span>}
                </div>
              )}
              {data ? (
                <div role="img" aria-label={sel ? `Scatter plot with ${data.n} data points and ${sel.name} fit curve (R²=${sel.adjR2.toFixed(4)})` : `Scatter plot with ${data.n} data points`}>
                  <ResponsiveContainer width="100%" height={370}>
                    <ComposedChart margin={{ top: 10, right: 20, bottom: 40, left: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis dataKey="x" type="number" stroke="#9CA3AF" tick={{ fontSize: 11 }}
                        scale={logX ? "log" : "auto"}
                        domain={logX ? (chartDataMemo.xDomain || ['auto', 'auto']) : axisTicks.x ? [axisTicks.x[0], axisTicks.x[axisTicks.x.length - 1]] : ['auto', 'auto']}
                        ticks={logX ? undefined : axisTicks.x}
                        allowDataOverflow
                        tickFormatter={logX ? (v) => v >= 1 ? v.toFixed(0) : v.toPrecision(2) : linearTickFmt}
                        label={{ value: data?.headers[0] || 'x', position: 'bottom', offset: 20, fill: '#9CA3AF', fontSize: 12 }} />
                      <YAxis stroke="#9CA3AF" tick={{ fontSize: 11 }}
                        scale={logY ? "log" : "auto"}
                        domain={logY ? ['auto', 'auto'] : axisTicks.y ? [axisTicks.y[0], axisTicks.y[axisTicks.y.length - 1]] : ['auto', 'auto']}
                        ticks={logY ? undefined : axisTicks.y}
                        allowDataOverflow={logY}
                        tickFormatter={logY ? (v) => v >= 1 ? v.toFixed(0) : v.toPrecision(2) : linearTickFmt}
                        label={{ value: data?.headers[1] || 'y', angle: -90, position: 'insideLeft', offset: -5, fill: '#9CA3AF', fontSize: 12 }} />
                      <Tooltip contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151', borderRadius: 8, fontSize: 11 }} />
                      {showCI && chartDataMemo.fp.length > 0 && chartDataMemo.fp[0]?.bandUpper != null && (
                        <>
                          <Line data={chartDataMemo.fp} dataKey="bandUpper" stroke="#3B82F6" strokeWidth={1} strokeDasharray="4 3" dot={false} isAnimationActive={false} opacity={0.35} name="Mean CI upper" />
                          <Line data={chartDataMemo.fp} dataKey="bandLower" stroke="#3B82F6" strokeWidth={1} strokeDasharray="4 3" dot={false} isAnimationActive={false} opacity={0.35} name="Mean CI lower" />
                        </>
                      )}
                      {chartDataMemo.fp.length > 0 && <Line data={chartDataMemo.fp} dataKey="yFit" stroke="#3B82F6" strokeWidth={2.5} dot={false} name="Fit" isAnimationActive={false} />}
                      <Scatter data={chartDataMemo.dp} dataKey="y" fill="#F59E0B" name="Data" r={4} isAnimationActive={false} />
                    </ComposedChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <div className="h-72 flex items-center justify-center text-gray-500 text-center">
                  <div><div className="text-4xl mb-3">📊</div><p className="text-sm">Paste data or upload CSV to begin</p></div>
                </div>
              )}
            </section>

            {sel && (
              <div className="flex gap-2 flex-wrap" role="toolbar" aria-label="Display and export controls">
                <button onClick={() => setShowResiduals(!showResiduals)} aria-pressed={showResiduals}
                  className="text-xs px-3 py-1.5 bg-gray-800 hover:bg-gray-700 rounded text-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500">
                  {showResiduals ? "Hide" : "Show"} Residuals</button>
                <button onClick={() => setShowCI(!showCI)} aria-pressed={showCI}
                  className="text-xs px-3 py-1.5 bg-gray-800 hover:bg-gray-700 rounded text-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500">
                  {showCI ? "Hide" : "Show"} Uncertainties</button>
                <button onClick={handleExportSVG} className="text-xs px-3 py-1.5 bg-gray-800 hover:bg-gray-700 rounded text-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500">Export SVG</button>
                <button onClick={handleExportCSV} className="text-xs px-3 py-1.5 bg-gray-800 hover:bg-gray-700 rounded text-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500">Export CSV</button>
                <button onClick={() => setProModalOpen(true)} className="flex items-center gap-1 text-xs px-3 py-1.5 bg-gray-800 hover:bg-gray-700 rounded text-gray-300 border border-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500"><span className="text-amber-400">🔒</span> Export PDF <span className="text-[10px] font-semibold bg-amber-500/20 text-amber-400 px-1.5 py-0.5 rounded-full leading-none">PRO</span></button>
                <button onClick={() => setProModalOpen(true)} className="flex items-center gap-1 text-xs px-3 py-1.5 bg-gray-800 hover:bg-gray-700 rounded text-gray-300 border border-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500"><span className="text-amber-400">🔒</span> Export PNG 300dpi <span className="text-[10px] font-semibold bg-amber-500/20 text-amber-400 px-1.5 py-0.5 rounded-full leading-none">PRO</span></button>
                <button onClick={() => copyParams(sel)}
                  className="text-xs px-3 py-1.5 bg-gray-800 hover:bg-gray-700 rounded text-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500">
                  {copied === sel.name ? '✓ Copied!' : 'Copy Params'}
                </button>
              </div>
            )}

            {showResiduals && residualData.length > 0 && (
              <section className="bg-gray-900 rounded-lg p-4 border border-gray-800" aria-label="Residual plot">
                <h3 className="text-xs text-gray-400 mb-2 font-medium">Residuals</h3>
                <ResponsiveContainer width="100%" height={140}>
                  <ComposedChart data={residualData} margin={{ top: 5, right: 20, bottom: 20, left: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="x" type="number" stroke="#9CA3AF" tick={{ fontSize: 10 }}
                      ticks={logX ? undefined : axisTicks.x}
                      domain={logX ? ['auto', 'auto'] : axisTicks.x ? [axisTicks.x[0], axisTicks.x[axisTicks.x.length - 1]] : ['auto', 'auto']}
                      tickFormatter={logX ? undefined : linearTickFmt} />
                    <YAxis stroke="#9CA3AF" tick={{ fontSize: 10 }}
                      tickFormatter={linearTickFmt} />
                    <Scatter dataKey="residual" fill="#EF4444" r={3} />
                    <Line dataKey="zero" stroke="#6B7280" strokeDasharray="5 5" dot={false} isAnimationActive={false} />
                  </ComposedChart>
                </ResponsiveContainer>
              </section>
            )}

            {/* Model details + CI */}
            {sel && (
              <section className="bg-gray-900 rounded-lg p-4 border border-gray-800" aria-label="Model details">
                {/* Quality badge */}
                <div className="flex items-center gap-2 mb-3 flex-wrap">
                  {sel.quality && (
                    <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium ${sel.quality.bg} ${sel.quality.color}`}
                      role="status">
                      <span className="font-bold">{sel.quality.icon}</span> {sel.quality.label}
                      {sel.quality.label === "Unreliable" && <span className="text-gray-500 font-normal ml-1">— do not use for publication</span>}
                    </span>
                  )}
                  {selectedModel === 0 && (
                    <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium bg-blue-500/15 text-blue-300">
                      ★ Best by AICc
                    </span>
                  )}
                </div>

                <div className="flex items-start justify-between mb-3">
                  <div>
                    <h3 className="font-semibold text-white text-lg">{sel.name}</h3>
                    <p className="text-blue-300 font-mono text-sm mt-0.5">{sel.equation}</p>
                  </div>
                  <div className="text-right">
                    <div className={`text-2xl font-bold font-mono ${sel.adjR2 > 0.99 ? 'text-green-400' : sel.adjR2 > 0.95 ? 'text-yellow-400' : sel.adjR2 < 0 ? 'text-red-500' : 'text-red-400'}`}>
                      {sel.adjR2 < 0 ? sel.adjR2.toFixed(3) : (sel.adjR2 * 100).toFixed(2) + "%"}
                    </div>
                    <div className="text-xs text-gray-500">{sel.adjR2 < 0 ? "adj. R² (negative!)" : "adj. R²"}</div>
                  </div>
                </div>

                {/* Model note */}
                {MODEL_NOTES[sel.family] && (
                  <div className="text-xs text-blue-400/80 bg-blue-500/10 rounded px-2 py-1.5 mb-3">
                    ℹ {MODEL_NOTES[sel.family]}
                  </div>
                )}

                {/* Parameters with CI */}
                <div className="overflow-x-auto">
                  <table className="w-full text-xs" aria-label="Fitted parameters">
                    <thead>
                      <tr className="text-gray-400 border-b border-gray-800">
                        <th className="text-left py-1 pr-3">Parameter</th>
                        <th className="text-right py-1 pr-3">Value</th>
                        {showCI && sel.stdErrors && <th className="text-right py-1 pr-3">Std Error</th>}
                        {showCI && sel.ci95 && <th className="text-right py-1">≈95% CI</th>}
                      </tr>
                    </thead>
                    <tbody>
                      {sel.paramNames.map((name, i) => (
                        <tr key={i} className="border-b border-gray-800/50">
                          <td className="py-1.5 pr-3 text-gray-300 font-medium">{name}</td>
                          <td className="py-1.5 pr-3 text-right font-mono text-white">{fmt(sel.params[i])}</td>
                          {showCI && sel.stdErrors && (
                            <td className="py-1.5 pr-3 text-right font-mono text-gray-400">
                              {isFinite(sel.stdErrors[i]) ? `±${fmt(sel.stdErrors[i])}` : "—"}
                            </td>
                          )}
                          {showCI && sel.ci95 && (
                            <td className="py-1.5 text-right font-mono text-gray-400">
                              {isFinite(sel.ci95[i][0]) ? `[${fmt(sel.ci95[i][0])}, ${fmt(sel.ci95[i][1])}]` : "—"}
                            </td>
                          )}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {showCI && sel.stdErrors && (
                    <p className="text-xs text-gray-400 mt-1.5 italic">
                      Shaded bands show ≈95% confidence interval of the mean fit (analytical delta-method approximation).
                      {sel.dof < 8 && <span className="text-yellow-500 not-italic"> Low DOF ({sel.dof}) — uncertainties may be unreliable.</span>}
                    </p>
                  )}
                </div>

                {/* Metrics row */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mt-3" aria-label="Fit statistics">
                  <div className="bg-gray-800 rounded p-2">
                    <div className="text-xs text-gray-400">AICc</div>
                    <div className="text-sm font-mono text-white">{sel.aicc.toFixed(1)}</div>
                  </div>
                  <div className="bg-gray-800 rounded p-2">
                    <div className="text-xs text-gray-400">ΔAICc</div>
                    <div className={`text-sm font-mono ${sel.deltaAicc < 2 ? 'text-green-400' : sel.deltaAicc < 10 ? 'text-yellow-400' : 'text-red-400'}`}>
                      {sel.deltaAicc.toFixed(1)}
                    </div>
                  </div>
                  <div className="bg-gray-800 rounded p-2">
                    <div className="text-xs text-gray-400">Akaike Weight</div>
                    <div className="text-sm font-mono text-white">{(sel.akaikeWeight * 100).toFixed(1)}%</div>
                  </div>
                  <div className="bg-gray-800 rounded p-2">
                    <div className="text-xs text-gray-400">DOF</div>
                    <div className="text-sm font-mono text-white">{sel.dof}</div>
                  </div>
                </div>

                {/* Interpretation */}
                <div className="mt-2 text-xs text-gray-400">
                  {sel.deltaAicc === 0 && `Best model. Akaike weight ${(sel.akaikeWeight * 100).toFixed(0)}% — `}
                  {sel.deltaAicc === 0 && sel.akaikeWeight > 0.9 && "strong evidence — best among tested models."}
                  {sel.deltaAicc === 0 && sel.akaikeWeight <= 0.9 && sel.akaikeWeight > 0.5 && "moderate evidence; consider alternatives."}
                  {sel.deltaAicc === 0 && sel.akaikeWeight <= 0.5 && "weak evidence — multiple models are plausible."}
                  {sel.deltaAicc > 0 && sel.deltaAicc < 2 && "Essentially equivalent to best model (ΔAICc < 2)."}
                  {sel.deltaAicc >= 2 && sel.deltaAicc < 10 && "Some support, but best model is considerably better."}
                  {sel.deltaAicc >= 10 && "No empirical support vs. best model (ΔAICc > 10)."}
                </div>

                {/* Warnings */}
                {sel.warnings && sel.warnings.length > 0 && (
                  <div className="mt-3 space-y-1" role="alert">
                    {sel.warnings.map((w, i) => (
                      <div key={i} className="text-xs text-yellow-500 bg-yellow-500/10 rounded px-2 py-1">⚠ {w}</div>
                    ))}
                  </div>
                )}
              </section>
            )}
          </main>
        </div>

        <footer className="mt-6 text-center text-xs text-gray-500">
          CurveFit · 25+ models + custom equations · Levenberg-Marquardt · AICc + Akaike weights · ≈95% CI + bands · Multi-start · No data leaves your browser
          <span className="mx-2">·</span>
          <a href="https://github.com/calyphi/curvefit" target="_blank" rel="noopener noreferrer" className="text-gray-500 hover:text-gray-300 transition-colors">GitHub</a>
        </footer>
      </div>

      <ProModal open={proModalOpen} onClose={() => setProModalOpen(false)} />
    </div>
  );
}
