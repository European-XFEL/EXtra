// Inject shared XFEL documentation header
document.addEventListener('DOMContentLoaded', function() {
  // Base URL for shared resources
  const baseUrl = 'https://raw.githubusercontent.com/European-XFEL/EXtra/refs/heads/header/docs';
  const headerUrl = `${baseUrl}/xfel-header-example.html`;
  const cssUrl = `${baseUrl}/css/xfel-header.css`;

  // Fetch and inject CSS as text to avoid MIME type issues
  fetch(cssUrl)
    .then(response => response.text())
    .then(css => {
      const style = document.createElement('style');
      style.textContent = css;
      document.head.appendChild(style);
    })
    .catch(error => console.warn('Error loading XFEL header CSS:', error));

  // Load and inject the HTML
  fetch(headerUrl)
    .then(response => {
      if (!response.ok) {
        console.warn('Failed to load XFEL header:', response.status);
        return null;
      }
      return response.text();
    })
    .then(html => {
      if (!html) return;

      // For MkDocs Material theme - inject before the main header
      const header = document.querySelector('.md-header');
      if (header) {
        const nav = document.createElement('div');
        nav.className = 'xfel-cross-nav';
        nav.innerHTML = html;
        header.parentNode.insertBefore(nav, header);
      } else {
        // Fallback: inject at the top of the body
        const nav = document.createElement('div');
        nav.className = 'xfel-cross-nav';
        nav.innerHTML = html;
        document.body.insertBefore(nav, document.body.firstChild);
      }
    })
    .catch(error => {
      console.warn('Error loading XFEL header:', error);
    });
});
