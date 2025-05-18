<h1>✨ Air Draw – Draw with Your Finger in the Air! 🎨</h1>

<p><strong>Air Draw</strong> is a computer vision-based app that lets you draw in the air using only your <strong>hand gestures</strong> and webcam. Move your <strong>index finger</strong> to draw, and change colors using <strong>different fingers</strong>!</p>

<h2>🔍 Features</h2>

<ul>
  <li>🖐️ Real-time <strong>hand and finger tracking</strong> using MediaPipe</li>
  <li>🖊️ Draw with your <strong>index finger</strong></li>
  <li>🌈 Change colors by lifting <strong>different fingers</strong></li>
  <li>💾 Save your drawings with the <code>S</code> key</li>
  <li>🖼️ View saved drawings in the <strong>Drawing Gallery</strong> with the <code>G</code> key</li>
  <li>🧼 Clear canvas with the <code>C</code> key</li>
</ul>

<h2>🎨 Color Controls</h2>

<table>
  <thead>
    <tr>
      <th>Finger Pattern</th>
      <th>Action / Color</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Only Index Finger Up</td><td>Draw with current color</td></tr>
    <tr><td>Index + Middle Finger Up</td><td>Change to <strong>Blue</strong></td></tr>
    <tr><td>Index + Ring Finger Up</td><td>Change to <strong>Green</strong></td></tr>
    <tr><td>Index + Pinky Up</td><td>Change to <strong>Red</strong></td></tr>
    <tr><td>All Fingers Up</td><td>Change to <strong>Black</strong></td></tr>
  </tbody>
</table>

<h2>🛠️ Tech Stack</h2>

<ul>
  <li>Python 🐍</li>
  <li>OpenCV 🎥</li>
  <li>MediaPipe 🤖</li>
  <li>NumPy 📊</li>
</ul>

<h2>🚀 How to Run</h2>

<pre><code>git clone https://github.com/Preetii18/Air-draw.git
cd Air-draw
pip install -r requirements.txt
python Air_draw.py
</code></pre>

<h2>🎮 Controls</h2>

<table>
  <thead>
    <tr><th>Key</th><th>Action</th></tr>
  </thead>
  <tbody>
    <tr><td><code>S</code></td><td>Save current drawing</td></tr>
    <tr><td><code>G</code></td><td>Open Drawing Gallery</td></tr>
    <tr><td><code>C</code></td><td>Clear the canvas</td></tr>
    <tr><td><code>ESC</code></td><td>Exit the app</td></tr>
  </tbody>
</table>

<h2>📁 Gallery</h2>

<p>All saved drawings are stored in the <code>gallery/</code> folder. Press <code>G</code> during the app to view your recent creations.</p>

<h2>📸 Preview</h2>

<p><em>Add a screenshot or demo GIF here!</em></p>

<h2>🙌 Contributions</h2>

<p>Feel free to fork and improve the project:</p>
<ul>
  <li>Add more color gestures 🎨</li>
  <li>Add shape tools ✏️</li>
  <li>Add GUI buttons 🖱️</li>
</ul>

<h2>📜 License</h2>

<p><strong>MIT License</strong> – Use freely and creatively!</p>
