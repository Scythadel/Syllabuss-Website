
document.addEventListener('DOMContentLoaded', function() {
    var overlay = document.getElementById('splash-overlay');
    requestAnimationFrame(function() {
        overlay.classList.add('fade-out');
    });
    overlay.addEventListener('transitionend', function() {
        overlay.parentNode.removeChild(overlay);
    });
});