import cv2
import numpy as np
from threading import Thread, Lock
import time
import queue
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from av import VideoFrame
import asyncio
import platform
import logging
from concurrent.futures import ThreadPoolExecutor

class VideoProcessor:
    def __init__(self, target_fps=30, target_resolution=(640, 480)):
        self.target_fps = target_fps
        self.target_resolution = target_resolution
        self.frame_time = 1.0 / target_fps
        self.last_frame_time = 0
        self.processing_executor = ThreadPoolExecutor(max_workers=2)
        
        # Initialize hardware acceleration if available
        self.hw_acceleration = self._setup_hw_acceleration()
        
    def _setup_hw_acceleration(self):
        """Setup hardware acceleration based on platform"""
        if platform.system() == 'Windows':
            # Try NVIDIA GPU acceleration
            try:
                return cv2.cudacodec.createVideoReader('')
            except:
                try:
                    # Try Intel Quick Sync
                    return cv2.videoio_registry.getBackendName(cv2.CAP_INTEL_MFX)
                except:
                    return None
        return None
        
    def preprocess_frame(self, frame):
        """Optimize frame for processing"""
        if frame is None:
            return None
            
        # Resize frame if needed
        current_h, current_w = frame.shape[:2]
        target_w, target_h = self.target_resolution
        
        if current_w != target_w or current_h != target_h:
            if self.hw_acceleration:
                # Use GPU acceleration for resizing
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(frame)
                gpu_frame = cv2.cuda.resize(gpu_frame, self.target_resolution)
                frame = gpu_frame.download()
            else:
                frame = cv2.resize(frame, self.target_resolution, 
                                 interpolation=cv2.INTER_LINEAR)
        
        return frame
        
    def should_process_frame(self):
        """Check if enough time has passed to process next frame"""
        current_time = time.time()
        if current_time - self.last_frame_time >= self.frame_time:
            self.last_frame_time = current_time
            return True
        return False

class VideoStreamTrack(MediaStreamTrack):
    kind = "video"
    
    def __init__(self, track, processor):
        super().__init__()
        self.track = track
        self.processor = processor
        self.frame_queue = queue.Queue(maxsize=2)
        self.lock = Lock()
        self._start_processing()
        
    def _start_processing(self):
        """Start background frame processing"""
        self.active = True
        self.process_thread = Thread(target=self._process_frames)
        self.process_thread.daemon = True
        self.process_thread.start()
        
    def _process_frames(self):
        """Background frame processing loop"""
        while self.active:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                if frame is not None and self.processor.should_process_frame():
                    # Process frame in thread pool
                    future = self.processor.processing_executor.submit(
                        self.processor.preprocess_frame, frame)
                    processed_frame = future.result()
                    
                    if processed_frame is not None:
                        # Convert to VideoFrame for WebRTC
                        video_frame = VideoFrame.from_ndarray(
                            processed_frame, format="bgr24"
                        )
                        video_frame.pts = int(time.time() * 1000)
                        video_frame.time_base = "ms"
                        
                        with self.lock:
                            self._current_frame = video_frame
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Frame processing error: {str(e)}")
                
    async def recv(self):
        """Receive the next frame"""
        if self.track:
            frame = await self.track.recv()
            # Add frame to processing queue if not full
            if not self.frame_queue.full():
                self.frame_queue.put(frame.to_ndarray())
            
            # Return processed frame if available
            with self.lock:
                if hasattr(self, '_current_frame'):
                    return self._current_frame
            
            # Return original frame if no processed frame available
            return frame
        else:
            return None

class RTCVideoStream:
    def __init__(self, target_fps=30, target_resolution=(640, 480)):
        self.processor = VideoProcessor(target_fps, target_resolution)
        self.peer_connections = set()
        self.local_tracks = set()
        
    async def create_offer(self, track):
        """Create WebRTC offer for video streaming"""
        pc = RTCPeerConnection()
        self.peer_connections.add(pc)
        
        # Create optimized video track
        video_track = VideoStreamTrack(track, self.processor)
        self.local_tracks.add(video_track)
        pc.addTrack(video_track)
        
        # Create offer
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        
        return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        
    async def process_answer(self, pc_index, answer):
        """Process WebRTC answer from client"""
        pc = list(self.peer_connections)[pc_index]
        await pc.setRemoteDescription(
            RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
        )
        
    async def stop(self):
        """Stop all WebRTC connections and cleanup"""
        # Close peer connections
        coros = [pc.close() for pc in self.peer_connections]
        await asyncio.gather(*coros)
        self.peer_connections.clear()
        
        # Stop local tracks
        for track in self.local_tracks:
            track.active = False
        self.local_tracks.clear()
        
        # Cleanup processor
        self.processor.processing_executor.shutdown()
