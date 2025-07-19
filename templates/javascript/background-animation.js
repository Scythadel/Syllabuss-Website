// background-animation.js
document.addEventListener('DOMContentLoaded', () => {
  // —— animated-gradient tweening ——
  const bg        = document.querySelector('.animated-background');
  const numSpots  = 8;     // how many “color spots” you’re randomizing
  const interval  = 4000;  // ms between picks
  console.log('bg element is', bg);
  // helper: smoothly animate one CSS var from→to
   function parseValue(str) {
    const m = str.trim().match(/^(-?[\d.]+)(.*)$/);
    return m
      ? { num: parseFloat(m[1]), unit: m[2] }
      : { num: 0, unit: '' };
  }

  // manual tween of a CSS variable over `duration` ms
  function tweenVar(prop, from, to, duration = interval) {
    const start = performance.now();
    const a     = parseValue(from);
    const b     = parseValue(to);

    function step(now) {
      const t = Math.min((now - start) / duration, 1);
      const current = a.num + (b.num - a.num) * t;
      bg.style.setProperty(prop, current + a.unit);
      if (t < 1) {
        requestAnimationFrame(step);
      }
    }

    requestAnimationFrame(step);
  }

  function randomizeBackground() {
    const style    = getComputedStyle(bg);
    // — angle
    const oldAng   = style.getPropertyValue('--angle') || '0deg';
    const newAng   = (Math.random()*360).toFixed(1) + 'deg';
    tweenVar('--angle', oldAng.trim(), newAng);

    // — each spot’s x/y
    for (let i=0; i<numSpots; i++) {
      const ox = style.getPropertyValue(`--spot${i}-x`) || '50%';
      const oy = style.getPropertyValue(`--spot${i}-y`) || '50%';
      const nx = (Math.random()*100).toFixed(1) + '%';
      const ny = (Math.random()*100).toFixed(1) + '%';
      tweenVar(`--spot${i}-x`, ox.trim(), nx);
      tweenVar(`--spot${i}-y`, oy.trim(), ny);
    }
  }

  // kick off and loop
  randomizeBackground();
  setInterval(randomizeBackground, interval);
});
