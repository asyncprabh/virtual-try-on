from flask import Blueprint, request, jsonify
from utils.video_stream import RTCVideoStream
from aiortc import RTCPeerConnection, RTCSessionDescription

webrtc = Blueprint('webrtc', __name__)

# Initialize video stream
video_stream = RTCVideoStream(target_fps=30, target_resolution=(640, 480))

@webrtc.route('/offer', methods=['POST'])
async def webrtc_offer():
    """Handle WebRTC offer from client"""
    try:
        params = await request.get_json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
        
        pc = RTCPeerConnection()
        pc_index = len(video_stream.peer_connections)
        
        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            if pc.connectionState == "failed":
                await pc.close()
        
        # Get answer for the offer
        answer = await video_stream.create_offer(offer)
        
        return jsonify(answer)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@webrtc.route('/answer', methods=['POST'])
async def webrtc_answer():
    """Handle WebRTC answer from client"""
    try:
        params = await request.get_json()
        pc_index = params.get('pc_index', 0)
        answer = {"sdp": params["sdp"], "type": params["type"]}
        
        await video_stream.process_answer(pc_index, answer)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@webrtc.route('/stream/start', methods=['POST'])
async def start_stream():
    """Start video streaming session"""
    try:
        # Get stream configuration from request
        config = request.get_json()
        fps = config.get('fps', 30)
        width = config.get('width', 640)
        height = config.get('height', 480)
        
        # Update video stream configuration
        video_stream.processor.target_fps = fps
        video_stream.processor.target_resolution = (width, height)
        
        return jsonify({
            'success': True,
            'message': 'Streaming session started',
            'config': {
                'fps': fps,
                'resolution': f"{width}x{height}"
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@webrtc.route('/stream/stop', methods=['POST'])
async def stop_stream():
    """Stop video streaming session"""
    try:
        await video_stream.stop()
        return jsonify({
            'success': True,
            'message': 'Streaming session stopped'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
