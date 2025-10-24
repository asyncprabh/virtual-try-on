class WebRTCClient {
    constructor(config = {}) {
        this.config = {
            fps: config.fps || 30,
            width: config.width || 640,
            height: config.height || 480,
            iceServers: config.iceServers || [
                { urls: ["stun:stun.l.google.com:19302"] }
            ]
        };
        
        this.peerConnection = null;
        this.localStream = null;
        this.videoElement = null;
        this.isStreaming = false;
    }
    
    async initialize(videoElement) {
        this.videoElement = videoElement;
        
        try {
            // Get user media with constraints
            const constraints = {
                video: {
                    width: { ideal: this.config.width },
                    height: { ideal: this.config.height },
                    frameRate: { ideal: this.config.fps }
                },
                audio: false
            };
            
            this.localStream = await navigator.mediaDevices.getUserMedia(constraints);
            this.videoElement.srcObject = this.localStream;
            
            // Initialize WebRTC connection
            this.peerConnection = new RTCPeerConnection({
                iceServers: this.config.iceServers
            });
            
            // Add tracks to peer connection
            this.localStream.getTracks().forEach(track => {
                this.peerConnection.addTrack(track, this.localStream);
            });
            
            // Handle ICE candidates
            this.peerConnection.onicecandidate = event => {
                if (event.candidate) {
                    // Send candidate to server if needed
                }
            };
            
            // Handle connection state changes
            this.peerConnection.onconnectionstatechange = () => {
                console.log("Connection state:", this.peerConnection.connectionState);
            };
            
            // Handle incoming tracks
            this.peerConnection.ontrack = event => {
                if (this.videoElement) {
                    this.videoElement.srcObject = event.streams[0];
                }
            };
            
            return true;
        } catch (error) {
            console.error("WebRTC initialization error:", error);
            throw error;
        }
    }
    
    async startStreaming() {
        if (!this.peerConnection) {
            throw new Error("WebRTC not initialized");
        }
        
        try {
            // Start stream on server
            const response = await fetch('/webrtc/stream/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(this.config)
            });
            
            if (!response.ok) {
                throw new Error("Failed to start stream on server");
            }
            
            // Create and send offer
            const offer = await this.peerConnection.createOffer();
            await this.peerConnection.setLocalDescription(offer);
            
            const offerResponse = await fetch('/webrtc/offer', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    sdp: offer.sdp,
                    type: offer.type
                })
            });
            
            const answer = await offerResponse.json();
            await this.peerConnection.setRemoteDescription(
                new RTCSessionDescription(answer)
            );
            
            this.isStreaming = true;
            return true;
        } catch (error) {
            console.error("Streaming error:", error);
            throw error;
        }
    }
    
    async stopStreaming() {
        try {
            // Stop stream on server
            await fetch('/webrtc/stream/stop', {
                method: 'POST'
            });
            
            // Close peer connection
            if (this.peerConnection) {
                this.peerConnection.close();
                this.peerConnection = null;
            }
            
            // Stop local stream
            if (this.localStream) {
                this.localStream.getTracks().forEach(track => track.stop());
                this.localStream = null;
            }
            
            this.isStreaming = false;
            return true;
        } catch (error) {
            console.error("Error stopping stream:", error);
            throw error;
        }
    }
    
    setVideoProcessor(processor) {
        this.videoProcessor = processor;
    }
    
    async updateStreamConfig(config) {
        Object.assign(this.config, config);
        
        if (this.isStreaming) {
            // Restart streaming with new config
            await this.stopStreaming();
            await this.startStreaming();
        }
    }
}
