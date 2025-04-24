
        // Toggle motion detection
        document.getElementById('toggle-motion').onclick = function() {
            fetch('/toggle_motion_detection', { method: 'POST' });
        };

        // Toggle between light and dark mode
        document.getElementById('toggle-theme').onclick = function() {
            document.body.classList.toggle('light-mode');
        };

        // Normal recording
        let isNormalRecording = false;
        document.getElementById('record-button').onclick = function() {
            if (!isNormalRecording) {
                // Start normal recording
                fetch('/start_recording', { method: 'POST' });
                document.getElementById('record-button').classList.add('recording');
                document.getElementById('record-button').textContent = 'Stop Normal Recording';
                isNormalRecording = true;

                // Disable motion recording if normal recording is started
                document.getElementById('motion-record-button').disabled = true;
                document.getElementById('lookout-button').disabled = true;
            } else {
                // Stop normal recording
                fetch('/stop_recording', { method: 'POST' });
                document.getElementById('record-button').classList.remove('recording');
                document.getElementById('record-button').textContent = 'Start Normal Recording';
                isNormalRecording = false;

                // Enable motion recording again if normal recording is stopped
                document.getElementById('motion-record-button').disabled = false;
                document.getElementById('lookout-button').disabled = false;
            }
        };

        // Motion detection recording
        let isMotionRecording = false;
        document.getElementById('motion-record-button').onclick = function() {
            if (!isMotionRecording) {
                // Start motion detection recording
                fetch('/start_motion_recording', { method: 'POST' });
                document.getElementById('motion-record-button').classList.add('recording');
                document.getElementById('motion-record-button').textContent = 'Stop Motion Detection Recording';
                isMotionRecording = true;

                // Disable normal recording if motion recording is started
                document.getElementById('record-button').disabled = true;
                document.getElementById('lookout-button').disabled = true;
            } else {
                // Stop motion detection recording
                fetch('/stop_motion_recording', { method: 'POST' });
                document.getElementById('motion-record-button').classList.remove('recording');
                document.getElementById('motion-record-button').textContent = 'Start Motion Detection Recording';
                isMotionRecording = false;

                // Enable normal recording again if motion detection recording is stopped
                document.getElementById('record-button').disabled = false;
                document.getElementById('lookout-button').disabled = false;
            }
        };

        // Lookout mode recording
        let isLookoutRecording = false;
        document.getElementById('lookout-button').onclick = function() {
            if (!isLookoutRecording) {
                // Start lookout recording
                fetch('/start_lookout', { method: 'POST' });
                document.getElementById('lookout-button').classList.add('recording');
                document.getElementById('lookout-button').textContent = 'Stop AI Lookout Mode';
                isLookoutRecording = true;

                // Disable other recordings
                document.getElementById('record-button').disabled = true;
                document.getElementById('motion-record-button').disabled = true;
            } else {
                // Stop lookout recording
                fetch('/stop_lookout', { method: 'POST' });
                document.getElementById('lookout-button').classList.remove('recording');
                document.getElementById('lookout-button').textContent = 'Start AI Lookout Mode';
                isLookoutRecording = false;

                // Enable other recordings
                document.getElementById('record-button').disabled = false;
                document.getElementById('motion-record-button').disabled = false;
            }
        };

        // Toggle lookout highlight
        function toggleLookoutHighlight() {
            fetch('/toggle_lookout_highlight', { method: 'POST' });
        }


const sliders = ['brightness', 'contrast'];

sliders.forEach(id => {
  const slider = document.getElementById(id);
  const label = document.getElementById(id + 'Val');

  slider.addEventListener('input', () => {
    // Update the value displayed next to the slider
    label.textContent = slider.value;

    // Send the updated values to the backend
    fetch('/update_video_settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        brightness: document.getElementById('brightness').value,
        contrast: document.getElementById('contrast').value,
      })
    });
  });
});

document.getElementById('video-size').addEventListener('input', function() {
    let videoFeed = document.querySelector('.video-feed');
    let videoSize = this.value;

    // Update video size and display the value
    videoFeed.style.maxWidth = videoSize + 'px';
    document.getElementById('videoSizeVal').textContent = videoSize;
});



document.getElementById('feature-toggle-form').addEventListener('submit', function (e) {
    e.preventDefault();

    const voice = document.getElementById('toggle-voice').checked ? 1 : 0;
    const notify = document.getElementById('toggle-notify').checked ? 1 : 0;

    fetch('/toggle_features', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ voice: voice, notify: notify })
    })
    .then(res => res.json())
    .then(data => {
        alert("Settings updated:\nVoice: " + data.voice + "\nNotify: " + data.notify);
    });
});
