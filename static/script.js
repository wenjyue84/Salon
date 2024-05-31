let totalTokensUsed = 0;

document.getElementById('chat-form').addEventListener('submit', function(event) {
    event.preventDefault();
    const question = document.getElementById('question').value;
    fetch('/ask', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: `question=${encodeURIComponent(question)}`
    })
    .then(response => response.json())
    .then(data => {
        const chatBox = document.getElementById('chat-box');
        chatBox.innerHTML += `<div class="message user"><strong>User:</strong> ${question}</div>`;
        chatBox.innerHTML += `<div class="message assistant"><strong>Assistant:</strong> ${data.answer}</div>`;
        document.getElementById('question').value = '';
        chatBox.scrollTop = chatBox.scrollHeight;

        totalTokensUsed += data.tokens_used;
        document.getElementById('tokenUsage').textContent = `Tokens Used: ${totalTokensUsed}`;
    })
    .catch(error => console.error('Error:', error));
});
