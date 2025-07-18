//This function is called as soon as the domain loads
document.addEventListener('DOMContentLoaded', function() {
    //getting the required document element
    var overlay = document.getElementById('splash-overlay');

    //fadeout triggered
    requestAnimationFrame(function() {
        overlay.classList.add('fade-out');
    });
    //remove overlay in the end so it doesnt block anything
    overlay.addEventListener('transitionend', function() {
        overlay.parentNode.removeChild(overlay);
    });
});