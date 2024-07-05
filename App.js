import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');

  const sendMessage = async () => {
    const response = await axios.post('http://localhost:5000/chat', { message: input });
    setMessages([...messages, { sender: 'user', text: input }, { sender: 'bot', text: response.data.response }]);
    setInput('');
  };

  return (
      <Container>
      <List>
        {messages.map((msg, index) => (
          <ListItem key={index} alignItems="flex-start">
            <ListItemText primary={msg.sender === 'user' ? "You" : "Bot"} secondary={msg.text} />
          </ListItem>
        ))}
      </List>
      <TextField
        label="Type your message"
        fullWidth
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
      />
      <Button variant="contained" color="primary" onClick={sendMessage}>
        Send
      </Button>
    </Container>
  );
}

export default App;
