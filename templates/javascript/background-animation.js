
document.addEventListener('DOMContentLoaded', () => {
  const background = document.querySelector('.animated-background');
  const numberOfSpots  = 8;

  const minimumInterval = 2000;
  const maximumInterval = 6000; // ms
  const baseSpeedFactor = 300;
  let changeInterval = maximumInterval;

  let lastMouseX, lastMouseY, lastMeasuredTime;
  let canAnimate = true;

  const throttleDelay = 50; //ms
  let animationCurrentVersion = 0;


  // helpers
  function parseValueFromStrToFloat(str) {
    const inpStr = str.trim().match(/^(-?[\d.]+)(.*)$/); //e.g 45deg -> inpStr[1]='45', inpStr[2]='deg'
    return inpStr
      ? { num: parseFloat(inpStr[1]), unit: inpStr[2] } //e.g { num: 45, unit: 'deg'}
      : { num: 0, unit: '' };
  }
  function easeOutCubic(time) {
    return 1 - Math.pow(1 - time, 3);
  }

  function tweenVar(propertyCSS, startVal, endVal, version) {
    const startTime = performance.now();
    const from = parseValueFromStrToFloat(startVal);
    const to = parseValueFromStrToFloat(endVal);

    function step(currentTimestamp) {
      if (version !== animationCurrentVersion) return;
      const t = Math.min((currentTimestamp - startTime) / changeInterval, 1);
      const timeNormalizedAndEased = easeOutCubic(t);
      const current = from.num + (to.num - from.num) * timeNormalizedAndEased;
      background.style.setProperty(propertyCSS, current + from.unit);
      if (t < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  function randomizeBackground(withTween = true) {
    animationCurrentVersion++;
    const version= animationCurrentVersion;
    const style = getComputedStyle(background);

    const oldAngle = style.getPropertyValue('--angle') || '0deg';
    const newAngle = (Math.random()*360).toFixed(1) + 'deg';
    if (withTween) tweenVar('--angle', oldAngle.trim(), newAngle, version);
    else background.style.setProperty(`--angle`, oldAngle.trim());
    
    for (let i=0; i<numberOfSpots; i++) {
      const oldx = style.getPropertyValue(`--spot${i}-x`) || '50%';
      const oldy = style.getPropertyValue(`--spot${i}-y`) || '50%';
      const newx = (Math.random()*100).toFixed(1) + '%';
      const newy = (Math.random()*100).toFixed(1) + '%';
      if (withTween) {
        tweenVar(`--spot${i}-x`, oldx.trim(), newx, version);
        tweenVar(`--spot${i}-y`, oldy.trim(), newy, version);
      } else {
        background.style.setProperty(`--spot${i}-x`, newx);
        background.style.setProperty(`--spot${i}-y`, newy);
      }
    }
  }


  randomizeBackground(false);
  document.addEventListener('mousemove', (e) => {
    const now = performance.now()
    if (lastMeasuredTime !== undefined) {
      const deltaTime = now - lastMeasuredTime;
      const deltaX = e.clientX - lastMouseX;
      const deltaY = e.clientY - lastMouseY;
      const distance = Math.hypot(deltaX, deltaY);
      const speed = distance / deltaTime;
      const intervalOfMovement = baseSpeedFactor / (speed || 0.01);
      changeInterval = Math.max(minimumInterval, Math.min(maximumInterval, intervalOfMovement));
    }
    lastMouseX = e.clientX;
    lastMouseY = e.clientY;
    lastMeasuredTime = now;

    if (!canAnimate) return;
    canAnimate = false;
    randomizeBackground(true);
    setTimeout(() => { canAnimate = true }, throttleDelay);
  });

});
