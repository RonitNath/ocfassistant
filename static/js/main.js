// DOM Elements
const navButtons = document.querySelectorAll('.nav-button');
const tabContents = document.querySelectorAll('.tab-content');
const chatInput = document.getElementById('chat-input');
const sendButton = document.getElementById('send-button');
const chatMessages = document.getElementById('chat-messages');

// WebSocket Connection
let socket;
let reconnectAttempts = 0;
const maxReconnectAttempts = 10;
const reconnectDelay = 2000; // Start with 2 second delay
let pingInterval;
let lastPongTime = 0;
let clientId = null;
let connectionStatus = 'disconnected';

// Initialize WebSocket connection
function initializeSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const socketUrl = `${protocol}//${window.location.host}/socket.io/?EIO=4&transport=websocket`;
    
    console.log('Initializing WebSocket connection...');
    
    // If there's an existing socket, clean it up
    if (socket) {
        clearInterval(pingInterval);
        socket.close();
    }
    
    // Socket.IO is loaded via a script tag in the HTML
    socket = io({
        reconnection: false, // We'll handle reconnection manually
        timeout: 20000,
        transports: ['websocket']
    });
    
    // Connection established
    socket.on('connect', () => {
        console.log('WebSocket connected!');
        connectionStatus = 'connected';
        reconnectAttempts = 0;
        clientId = socket.id;
        updateConnectionStatus('Connected');
        
        // Setup ping interval
        startPingInterval();
    });
    
    // Connection error
    socket.on('connect_error', (error) => {
        console.error('Connection error:', error);
        connectionStatus = 'error';
        updateConnectionStatus('Connection Error');
        handleReconnect();
    });
    
    // Disconnected
    socket.on('disconnect', (reason) => {
        console.log('WebSocket disconnected. Reason:', reason);
        connectionStatus = 'disconnected';
        updateConnectionStatus('Disconnected');
        clearInterval(pingInterval);
        handleReconnect();
    });
    
    // Handle server pong responses
    socket.on('pong', (data) => {
        lastPongTime = Date.now();
        console.log('Pong received:', data);
        const latency = Date.now() - data.received.timestamp;
        updateConnectionStatus(`Connected (Latency: ${latency}ms)`);
    });
    
    // Handle server status updates
    socket.on('status_update', (data) => {
        console.log('Status update:', data);
        updateDashboardMetrics(data);
    });
    
    // Additional event for connection response
    socket.on('connection_response', (data) => {
        console.log('Connection response:', data);
        clientId = data.client_id;
    });
}

// Send ping messages to server
function startPingInterval() {
    clearInterval(pingInterval);
    pingInterval = setInterval(() => {
        if (connectionStatus === 'connected') {
            sendPing();
            
            // Check if we haven't received a pong in a while
            const now = Date.now();
            if (lastPongTime > 0 && now - lastPongTime > 45000) {
                console.warn('No pong received in 45 seconds, reconnecting...');
                socket.disconnect();
                handleReconnect();
            }
        }
    }, 15000); // Send ping every 15 seconds
}

// Send ping with timestamp
function sendPing() {
    socket.emit('ping', {
        timestamp: Date.now(),
        client_id: clientId
    });
}

// Handle reconnection with exponential backoff
function handleReconnect() {
    if (reconnectAttempts < maxReconnectAttempts) {
        reconnectAttempts++;
        const delay = Math.min(reconnectDelay * Math.pow(1.5, reconnectAttempts - 1), 60000);
        updateConnectionStatus(`Reconnecting in ${Math.round(delay/1000)}s...`);
        
        console.log(`Reconnecting (attempt ${reconnectAttempts}/${maxReconnectAttempts}) in ${delay}ms`);
        
        setTimeout(() => {
            updateConnectionStatus('Reconnecting...');
            initializeSocket();
        }, delay);
    } else {
        updateConnectionStatus('Connection failed. Please refresh the page.');
        console.error('Max reconnection attempts reached');
    }
}

// Update connection status on the UI
function updateConnectionStatus(status) {
    // Add a connection status indicator to the header if it doesn't exist
    let statusElement = document.getElementById('connection-status');
    if (!statusElement) {
        statusElement = document.createElement('div');
        statusElement.id = 'connection-status';
        statusElement.className = 'connection-status';
        document.querySelector('.logo').appendChild(statusElement);
    }
    
    // Update status text and appearance
    statusElement.textContent = status;
    
    // Update class based on connection state
    statusElement.className = 'connection-status';
    if (status.includes('Connected')) {
        statusElement.classList.add('connected');
    } else if (status.includes('Reconnecting')) {
        statusElement.classList.add('reconnecting');
    } else {
        statusElement.classList.add('disconnected');
    }
}

// Update dashboard metrics when we receive status updates
function updateDashboardMetrics(data) {
    const metricsElement = document.getElementById('live-metrics');
    if (metricsElement) {
        metricsElement.innerHTML = `
            <div class="metric">
                <span class="metric-value">${data.connected_clients}</span>
                <span class="metric-label">Connected Clients</span>
            </div>
            <div class="metric">
                <span class="metric-value">${new Date(data.timestamp * 1000).toLocaleTimeString()}</span>
                <span class="metric-label">Last Update</span>
            </div>
        `;
    }
}

// Initialize WebSocket on page load
window.addEventListener('load', initializeSocket);

// Tab Navigation
navButtons.forEach(button => {
    button.addEventListener('click', () => {
        // Remove active class from all buttons and tabs
        navButtons.forEach(btn => btn.classList.remove('active'));
        tabContents.forEach(tab => tab.classList.remove('active'));
        
        // Add active class to clicked button and corresponding tab
        button.classList.add('active');
        const tabId = button.getAttribute('data-tab');
        document.getElementById(`${tabId}-tab`).classList.add('active');
    });
});

// Chat functionality
function sendMessage() {
    const message = chatInput.value.trim();
    if (message === '') return;
    
    // Add user message to chat
    addMessageToChat('user', message);
    
    // Clear input
    chatInput.value = '';
    
    // Simulate assistant response (this would be replaced with actual API call)
    simulateAssistantResponse();
}

function addMessageToChat(sender, content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    
    const paragraph = document.createElement('p');
    paragraph.textContent = content;
    
    messageContent.appendChild(paragraph);
    messageDiv.appendChild(messageContent);
    chatMessages.appendChild(messageDiv);
    
    // Scroll to bottom of chat
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function simulateAssistantResponse() {
    // Add loading indicator
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'message assistant';
    loadingDiv.innerHTML = '<div class="message-content"><p>...</p></div>';
    chatMessages.appendChild(loadingDiv);
    
    // Simulate API delay
    setTimeout(() => {
        // Remove loading indicator
        chatMessages.removeChild(loadingDiv);
        
        // Add assistant response
        addMessageToChat('assistant', 'This is a placeholder response. The backend API is not connected yet. When fully implemented, this will provide actual answers from the OCF Assistant.');
    }, 1500);
}

// Event listeners
sendButton.addEventListener('click', sendMessage);

chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Data tab functionality
document.getElementById('start-scrape').addEventListener('click', () => {
    const depth = document.getElementById('scrape-depth').value;
    alert(`Scraping initiated with depth ${depth}. This would trigger the backend API in the full implementation.`);
});

document.getElementById('simulate-scrape').addEventListener('click', () => {
    const depth = document.getElementById('scrape-depth').value;
    alert(`Simulation initiated with depth ${depth}. This would trigger the backend API in the full implementation.`);
});

document.getElementById('start-chunking').addEventListener('click', () => {
    alert('Chunking process initiated. This would trigger the backend API in the full implementation.');
});

document.getElementById('generate-embeddings').addEventListener('click', () => {
    alert('Embedding generation initiated. This would trigger the backend API in the full implementation.');
});

document.getElementById('upload-qdrant').addEventListener('click', () => {
    alert('Upload to Qdrant initiated. This would trigger the backend API in the full implementation.');
});

// Settings tab functionality
document.getElementById('save-settings').addEventListener('click', () => {
    alert('Settings saved. This would update the configuration in the full implementation.');
});

document.getElementById('reset-settings').addEventListener('click', () => {
    document.getElementById('embeddings-model').value = 'snowflake-arctic-embed2:latest';
    document.getElementById('chat-model').value = 'llama3.2:3b';
    document.getElementById('ollama-url').value = 'http://localhost:11434';
    document.getElementById('qdrant-url').value = 'http://localhost:6333';
    alert('Settings reset to default values.');
});

// Initialize with a welcome message
document.addEventListener('DOMContentLoaded', () => {
    // Nothing additional needed as the welcome message is in the HTML
});

// Dashboard functionality for connection testing
document.addEventListener('DOMContentLoaded', () => {
    const sendPingButton = document.getElementById('send-ping');
    const reconnectButton = document.getElementById('reconnect');
    const pingResults = document.getElementById('ping-results');
    const websocketStatus = document.getElementById('websocket-status');
    const lastPingElement = document.getElementById('last-ping');
    const clientIdElement = document.getElementById('client-id');
    
    if (sendPingButton) {
        sendPingButton.addEventListener('click', () => {
            if (connectionStatus === 'connected') {
                const timestamp = Date.now();
                const pingData = {
                    timestamp: timestamp,
                    client_id: clientId,
                    manual: true
                };
                
                // Add to results
                const resultItem = document.createElement('div');
                resultItem.textContent = `→ Ping sent: ${new Date(timestamp).toLocaleTimeString()}`;
                pingResults.prepend(resultItem);
                
                // Send ping
                socket.emit('ping', pingData);
                
                // Update UI
                lastPingElement.textContent = new Date(timestamp).toLocaleTimeString();
            } else {
                const resultItem = document.createElement('div');
                resultItem.textContent = `❌ Cannot send ping: Not connected`;
                pingResults.prepend(resultItem);
            }
        });
    }
    
    if (reconnectButton) {
        reconnectButton.addEventListener('click', () => {
            const resultItem = document.createElement('div');
            resultItem.textContent = `⟳ Manual reconnection initiated`;
            pingResults.prepend(resultItem);
            
            // Force disconnect if connected
            if (socket.connected) {
                socket.disconnect();
            }
            
            // Reset reconnection counter and initialize socket
            reconnectAttempts = 0;
            initializeSocket();
        });
    }
    
    // Update UI elements with connection status
    function updateDashboardStatus() {
        if (websocketStatus) {
            websocketStatus.textContent = connectionStatus.charAt(0).toUpperCase() + connectionStatus.slice(1);
        }
        
        if (clientIdElement && clientId) {
            clientIdElement.textContent = clientId;
        }
    }
    
    // Listen for the pong events to update dashboard
    if (socket) {
        socket.on('pong', (data) => {
            if (pingResults && data.received && data.received.manual) {
                const resultItem = document.createElement('div');
                const latency = Date.now() - data.received.timestamp;
                resultItem.textContent = `← Pong received: ${new Date(Date.now()).toLocaleTimeString()} (${latency}ms)`;
                pingResults.prepend(resultItem);
                
                // Limit the number of results shown
                if (pingResults.children.length > 10) {
                    pingResults.removeChild(pingResults.lastChild);
                }
            }
            
            updateDashboardStatus();
        });
        
        socket.on('connect', updateDashboardStatus);
        socket.on('disconnect', updateDashboardStatus);
    }
}); 