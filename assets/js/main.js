const menuButton=document.getElementById('menuButton');
const nav=document.getElementById('siteNav');
menuButton?.addEventListener('click',()=>{const open=nav.classList.toggle('open');menuButton.setAttribute('aria-expanded',String(open));});
nav?.querySelectorAll('a').forEach(a=>a.addEventListener('click',()=>nav.classList.remove('open')));
const observer=new IntersectionObserver(entries=>entries.forEach(e=>{if(e.isIntersecting){e.target.classList.add('visible');observer.unobserve(e.target);}}),{threshold:.08});
document.querySelectorAll('.reveal').forEach(el=>observer.observe(el));

// Muted autoplay is permitted by modern browsers. Start videos when they enter
// the viewport and pause them when they leave to avoid wasting bandwidth/CPU.
const missionVideos = document.querySelectorAll('.mission-video');
const videoObserver = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    const video = entry.target;
    if (entry.isIntersecting) {
      video.muted = true;
      const promise = video.play();
      if (promise && typeof promise.catch === 'function') promise.catch(() => {});
    } else {
      video.pause();
    }
  });
}, { threshold: 0.35 });
missionVideos.forEach((video) => videoObserver.observe(video));
