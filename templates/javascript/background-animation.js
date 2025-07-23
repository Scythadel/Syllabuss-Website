
document.addEventListener('DOMContentLoaded', () => {
  const background = document.querySelector('.animated-background');
  const numberOfSpots  = 8;
  const changeInterval  = 4000;  // ms between picks

  function parseValueFromStrToFloat(str) {
    const inpStr = str.trim().match(/^(-?[\d.]+)(.*)$/); //e.g 45deg -> inpStr[1]='45', inpStr[2]='deg'
    return inpStr
      ? { num: parseFloat(inpStr[1]), unit: inpStr[2] } //e.g { num: 45, unit: 'deg'}
      : { num: 0, unit: '' };
  }

  function tweenVar(propertyCSS, startVal, endVal) {
    const startTime = performance.now();
    const from = parseValueFromStrToFloat(startVal);
    const to = parseValueFromStrToFloat(endVal);

    function step(currentTimestamp) {
      const timeNormalized = Math.min((currentTimestamp - startTime) / changeInterval, 1);
      const current = from.num + (to.num - from.num) * timeNormalized;
      background.style.setProperty(propertyCSS, current + from.unit);
      if (timeNormalized < 1) {
        requestAnimationFrame(step);
      }
    }

    requestAnimationFrame(step);
  }

  function randomizeBackground() {
    const style = getComputedStyle(background);
    const oldAngle = style.getPropertyValue('--angle') || '0deg';
    const newAngle = (Math.random()*360).toFixed(1) + 'deg';
    tweenVar('--angle', oldAngle.trim(), newAngle);

    for (let i=0; i<numberOfSpots; i++) {
      const oldx = style.getPropertyValue(`--spot${i}-x`) || '50%';
      const oldy = style.getPropertyValue(`--spot${i}-y`) || '50%';
      const newx = (Math.random()*100).toFixed(1) + '%';
      const newy = (Math.random()*100).toFixed(1) + '%';
      tweenVar(`--spot${i}-x`, oldx.trim(), newx);
      tweenVar(`--spot${i}-y`, oldy.trim(), newy);
    }
  }

  randomizeBackground();
  setInterval(randomizeBackground, changeInterval);
});
