import { createCanvas } from 'canvas';
import { writeFileSync } from 'fs';

function drawCurveChart(ctx, x, y, w, h) {
  // Draw scatter points + fitted curve for visual flair
  const pts = [
    [0.05, 0.85], [0.12, 0.72], [0.20, 0.55], [0.30, 0.38],
    [0.42, 0.25], [0.55, 0.18], [0.68, 0.22], [0.78, 0.35],
    [0.88, 0.55], [0.95, 0.75],
  ];

  // Grid lines
  ctx.strokeStyle = 'rgba(100, 116, 139, 0.2)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const gy = y + (h * i) / 4;
    ctx.beginPath(); ctx.moveTo(x, gy); ctx.lineTo(x + w, gy); ctx.stroke();
    const gx = x + (w * i) / 4;
    ctx.beginPath(); ctx.moveTo(gx, y); ctx.lineTo(gx, y + h); ctx.stroke();
  }

  // Fitted curve (smooth bezier)
  ctx.strokeStyle = '#22d3ee';
  ctx.lineWidth = 3;
  ctx.beginPath();
  const curvePoints = [];
  for (let t = 0; t <= 1; t += 0.01) {
    const cy = 0.85 - 0.67 * Math.sin(Math.PI * t) * (1 - 0.3 * Math.cos(2 * Math.PI * t));
    curvePoints.push([x + t * w, y + cy * h]);
  }
  ctx.moveTo(curvePoints[0][0], curvePoints[0][1]);
  for (const [cx, cy] of curvePoints) ctx.lineTo(cx, cy);
  ctx.stroke();

  // Data points (scatter)
  ctx.fillStyle = '#f59e0b';
  for (const [px, py] of pts) {
    const dx = x + px * w + (Math.random() - 0.5) * 8;
    const dy = y + py * h + (Math.random() - 0.5) * 12;
    ctx.beginPath();
    ctx.arc(dx, dy, 5, 0, Math.PI * 2);
    ctx.fill();
  }
}

// ── OG Image for CurveFit ──
function genCurveFit() {
  const W = 1200, H = 630;
  const canvas = createCanvas(W, H);
  const ctx = canvas.getContext('2d');

  // Background gradient
  const grad = ctx.createLinearGradient(0, 0, W, H);
  grad.addColorStop(0, '#030712');
  grad.addColorStop(1, '#0f172a');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, W, H);

  // Brand: Calyphi (small top)
  ctx.fillStyle = '#22d3ee';
  ctx.font = 'bold 20px sans-serif';
  ctx.fillText('Calyphi', 60, 60);

  // Title
  ctx.fillStyle = '#ffffff';
  ctx.font = 'bold 64px sans-serif';
  ctx.fillText('CurveFit', 60, 160);

  // Subtitle
  ctx.fillStyle = '#94a3b8';
  ctx.font = '28px sans-serif';
  ctx.fillText('Scientific Curve Fitting', 60, 210);
  ctx.fillText('Instant · Accurate · Private', 60, 248);

  // Features
  ctx.fillStyle = '#64748b';
  ctx.font = '20px sans-serif';
  const features = ['25+ Models', 'Levenberg-Marquardt', 'AICc Ranking', '95% CI Bands'];
  features.forEach((f, i) => {
    ctx.fillStyle = '#22d3ee';
    ctx.fillText('▸', 60, 320 + i * 34);
    ctx.fillStyle = '#cbd5e1';
    ctx.fillText(f, 82, 320 + i * 34);
  });

  // Bottom tag
  ctx.fillStyle = '#475569';
  ctx.font = '18px sans-serif';
  ctx.fillText('calyphi.com/app — Free, no sign-up, 100% in-browser', 60, 580);

  // Chart on right side
  drawCurveChart(ctx, 660, 80, 480, 460);

  writeFileSync('public/og-curvefit.png', canvas.toBuffer('image/png'));
  console.log('✓ og-curvefit.png');
}

// ── OG Image for Calyphi brand ──
function genCalyphi() {
  const W = 1200, H = 630;
  const canvas = createCanvas(W, H);
  const ctx = canvas.getContext('2d');

  const grad = ctx.createLinearGradient(0, 0, W, H);
  grad.addColorStop(0, '#030712');
  grad.addColorStop(1, '#0f172a');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, W, H);

  // Brand name large
  ctx.fillStyle = '#22d3ee';
  ctx.font = 'bold 80px sans-serif';
  ctx.fillText('Calyphi', 60, 200);

  // Tagline
  ctx.fillStyle = '#ffffff';
  ctx.font = '36px sans-serif';
  ctx.fillText('Physics-driven tools for science', 60, 270);

  // Subtitle
  ctx.fillStyle = '#94a3b8';
  ctx.font = '24px sans-serif';
  ctx.fillText('Open, browser-first tools for researchers.', 60, 330);
  ctx.fillText('Rigorous statistics. Zero data collection. No subscriptions.', 60, 365);

  // Products
  ctx.font = 'bold 22px sans-serif';
  ctx.fillStyle = '#22d3ee';
  ctx.fillText('CurveFit', 60, 440);
  ctx.fillStyle = '#475569';
  ctx.font = '22px sans-serif';
  ctx.fillText('   ·   SimFit   ·   KinetiQ', 168, 440);

  // Bottom
  ctx.fillStyle = '#475569';
  ctx.font = '18px sans-serif';
  ctx.fillText('calyphi.com', 60, 580);

  // Decorative chart on right
  drawCurveChart(ctx, 700, 120, 420, 380);

  writeFileSync('public/og-calyphi.png', canvas.toBuffer('image/png'));
  console.log('✓ og-calyphi.png');
}

genCurveFit();
genCalyphi();
