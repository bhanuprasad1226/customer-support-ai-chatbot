const messages = document.getElementById('messages');
const input = document.getElementById('msg');
const sendBtn = document.getElementById('sendBtn');

function appendMessage(text, who) {
  const d = document.createElement('div');
  d.className = 'msg ' + who;
  d.innerHTML = text.replace(/\n/g, '<br>')

  messages.appendChild(d);
  messages.scrollTop = messages.scrollHeight;
}

async function sendMessage() {
  const txt = input.value.trim();
  if (!txt) return;

  appendMessage(txt, 'user');
  input.value = '';

  const t = document.createElement('div');
  t.className = 'msg bot';
  t.id = 'typing';
  t.innerText = 'Bot is typing...';
  messages.appendChild(t);
  messages.scrollTop = messages.scrollHeight;

  try {
    const res = await fetch('/api/message', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: txt })
    });

    const data = await res.json();
    const typ = document.getElementById('typing');
    if (typ) typ.remove();

    appendMessage(data.reply || 'Sorry, something went wrong.', 'bot');
  } catch (e) {
    const typ = document.getElementById('typing');
    if (typ) typ.remove();
    appendMessage('Server error. Try again.', 'bot');
    console.error(e);
  }
}

input.addEventListener('keydown', (e) => { if (e.key === 'Enter') sendMessage(); });
sendBtn.addEventListener('click', sendMessage);