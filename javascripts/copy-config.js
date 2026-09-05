document.addEventListener('DOMContentLoaded', function() {
  // Find all tables in the presets page
  const tables = document.querySelectorAll('table');

  tables.forEach(table => {
    const rows = table.querySelectorAll('tbody tr');

    rows.forEach(row => {
      const firstCell = row.querySelector('td:first-child');
      if (firstCell) {
        const codeElement = firstCell.querySelector('code');
        if (codeElement) {
          // Create copy button icon with SVG
          const copyIcon = document.createElement('span');
          copyIcon.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="16" height="16" style="vertical-align: middle;"><path fill="currentColor" d="M19,21H8V7H19M19,5H8A2,2 0 0,0 6,7V21A2,2 0 0,0 8,23H19A2,2 0 0,0 21,21V7A2,2 0 0,0 19,5M16,1H4A2,2 0 0,0 2,3V17H4V3H16V1Z"/></svg>`;
          copyIcon.style.marginLeft = '8px';
          copyIcon.style.opacity = '0.5';
          copyIcon.style.cursor = 'pointer';
          copyIcon.style.display = 'inline-block';
          copyIcon.title = 'Copy to clipboard';

          // Add icon after code element
          firstCell.appendChild(copyIcon);

          // Add copy functionality
          const copyFunction = function() {
            const text = codeElement.textContent;
            navigator.clipboard.writeText(text).then(() => {
              // Visual feedback - change to checkmark
              copyIcon.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="16" height="16" style="vertical-align: middle;"><path fill="currentColor" d="M21,7L9,19L3.5,13.5L4.91,12.09L9,16.17L19.59,5.59L21,7Z"/></svg>`;
              copyIcon.style.opacity = '1';
              setTimeout(() => {
                copyIcon.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="16" height="16" style="vertical-align: middle;"><path fill="currentColor" d="M19,21H8V7H19M19,5H8A2,2 0 0,0 6,7V21A2,2 0 0,0 8,23H19A2,2 0 0,0 21,21V7A2,2 0 0,0 19,5M16,1H4A2,2 0 0,0 2,3V17H4V3H16V1Z"/></svg>`;
                copyIcon.style.opacity = '0.5';
              }, 1000);
            });
          };

          copyIcon.addEventListener('click', copyFunction);
          codeElement.style.cursor = 'pointer';
          codeElement.addEventListener('click', copyFunction);
        }
      }
    });
  });
});
