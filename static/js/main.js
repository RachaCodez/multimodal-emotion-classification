// Basic enhancements: validation, file name preview, simple Chart.js hooks

document.addEventListener('DOMContentLoaded', function () {
  const audioInput = document.getElementById('audio_file');
  if (audioInput) {
    audioInput.addEventListener('change', function (e) {
      const fileName = e.target.files[0]?.name || 'No file chosen';
      const holder = document.getElementById('audio-filename');
      if (holder) holder.textContent = fileName;
    });
  }

  const imageInput = document.getElementById('image_file');
  if (imageInput) {
    imageInput.addEventListener('change', function (e) {
      const fileName = e.target.files[0]?.name || 'No file chosen';
      const holder = document.getElementById('image-filename');
      if (holder) holder.textContent = fileName;
    });
  }

  // Bootstrap validation
  const forms = document.querySelectorAll('.needs-validation');
  Array.from(forms).forEach(form => {
    form.addEventListener('submit', event => {
      if (!form.checkValidity()) {
        event.preventDefault();
        event.stopPropagation();
      }
      form.classList.add('was-validated');
    }, false);
  });
});

