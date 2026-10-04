// Lightweight replacements for the Privasis sidebar and figure lightbox.
const sidebar = document.querySelector('.sidebar-nav');
const intro = document.getElementById('intro');
const links = [...sidebar.querySelectorAll('a')];
function updateNavigation() {
  sidebar.classList.toggle('is-visible', intro.getBoundingClientRect().top <= 150);
  let active = null;
  links.forEach(link => {
    if (document.querySelector(link.hash).getBoundingClientRect().top <= 160) active = link;
  });
  if (window.scrollY + window.innerHeight >= document.documentElement.scrollHeight - 4) {
    active = links[links.length - 1];
  }
  links.forEach(link => {
    if (link === active) link.setAttribute('aria-current', 'location');
    else link.removeAttribute('aria-current');
  });
}
window.addEventListener('scroll', updateNavigation, { passive: true });
window.addEventListener('resize', updateNavigation);
updateNavigation();
const dialog = document.getElementById('figure-dialog');
const expanded = document.getElementById('expanded-figure');
document.querySelectorAll('.figure-link').forEach(link => {
  link.addEventListener('click', event => {
    if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey || !dialog.showModal) return;
    event.preventDefault();
    expanded.src = link.href;
    expanded.alt = link.querySelector('img').alt;
    dialog.showModal();
  });
});
document.getElementById('close-figure').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', event => { if (event.target === dialog) dialog.close(); });
document.getElementById('copy-citation').addEventListener('click', async () => {
  const citation = document.getElementById('bibtex');
  const status = document.getElementById('copy-status');
  try {
    await navigator.clipboard.writeText(citation.textContent);
    status.textContent = 'BibTeX copied to clipboard.';
  } catch {
    const range = document.createRange();
    range.selectNodeContents(citation);
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    status.textContent = 'Citation selected. Press Ctrl+C or ⌘C to copy.';
  }
});
