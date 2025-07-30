document.addEventListener('DOMContentLoaded', () => {
    // This will be from the server
    const alumniData = [
      { name: 'Big D', studied: 'Gooning', current: 'Jizzler', picUrl: 'https://example.com/filip.jpg' },
      { name: 'LOL', studied: 'Physics', current: 'Data Science', picUrl: 'https://example.com/joel.jpg' }
    ];
    const TOTAL_SLOTS = 8;

    const resultsPanel = document.querySelector('.results-panel');
    const headerRow = document.querySelector('.results-header');
    resultsPanel.innerHTML = '';
    resultsPanel.appendChild(headerRow);
    for (let i = 0; i < TOTAL_SLOTS; i++) {
      if (i < alumniData.length) {
        // the logic will be changed
        const alumn = alumniData[i];

        const row = document.createElement('div');
        row.className = 'profile-row';

        row.innerHTML = `
          <div class="profile-col">
            <div class="profile-pic" style="background-image:url('${alumn.picUrl}')"></div>
          </div>
          <div class="name-col">${alumn.name}</div>
          <div class="studied-col">${alumn.studied}</div>
          <div class="current-col">${alumn.current}</div>
        `;
        resultsPanel.appendChild(row);

      } else {
        const placeholder = document.createElement('div');
        placeholder.className = 'placeholder-row';
        placeholder.textContent = 'Profile';
        resultsPanel.appendChild(placeholder);
      }
    }


    document.querySelectorAll('.quick-item').forEach(btn => {
      btn.addEventListener('click', () => {
        document.getElementById('searchInput').value = btn.textContent;
      });
    });
});
