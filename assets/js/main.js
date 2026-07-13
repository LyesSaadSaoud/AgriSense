const menuButton=document.getElementById('menuButton');
const nav=document.getElementById('siteNav');
menuButton?.addEventListener('click',()=>{const open=nav.classList.toggle('open');menuButton.setAttribute('aria-expanded',String(open));});
nav?.querySelectorAll('a').forEach(a=>a.addEventListener('click',()=>nav.classList.remove('open')));

const observer=new IntersectionObserver(entries=>entries.forEach(e=>{if(e.isIntersecting){e.target.classList.add('visible');observer.unobserve(e.target);}}),{threshold:.12});
document.querySelectorAll('.reveal').forEach(el=>observer.observe(el));

async function attachVideos(){
  const cards=[...document.querySelectorAll('[data-video]')];
  for(const card of cards){
    const src=card.dataset.video;
    const video=card.querySelector('video');
    try{
      const res=await fetch(src,{method:'HEAD',cache:'no-store'});
      if(res.ok){video.src=src;card.classList.add('is-ready');}
    }catch(_){/* local preview or missing file: keep polished fallback */}
  }
}
attachVideos();
