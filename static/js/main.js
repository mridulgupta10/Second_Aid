// main.js - Second-Aid frontend utilities
// AOS (scroll animation) initialization
document.addEventListener('DOMContentLoaded', function () {
    if (typeof AOS !== 'undefined') {
        AOS.init({ duration: 800, once: true });
    }
});
