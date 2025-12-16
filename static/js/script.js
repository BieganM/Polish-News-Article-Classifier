function showTab(tabName) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => {
        c.classList.remove('active');
        c.style.animation = 'none';
    });

    document.querySelector(`[onclick="showTab('${tabName}')"]`).classList.add('active');
    
    const activeContent = document.getElementById(`${tabName}-tab`);
    activeContent.classList.add('active');
    void activeContent.offsetWidth;
    activeContent.style.animation = 'slideIn 0.5s forwards';
}
