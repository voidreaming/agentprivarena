// Native details supports mouse, touch, and keyboard navigation.
const researchSwitcher = document.querySelector('.research-switcher');
if (researchSwitcher) {
  document.addEventListener('click', event => {
    if (!researchSwitcher.contains(event.target)) researchSwitcher.open = false;
  });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && researchSwitcher.open) {
      researchSwitcher.open = false;
      researchSwitcher.querySelector('summary').focus();
    }
  });
}
