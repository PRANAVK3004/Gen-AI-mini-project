import React, { useState, useEffect, useRef } from 'react';
import { Mic, MicOff, PhoneOff } from 'lucide-react';

const VoiceAssistant = () => {
  const [conversation, setConversation] = useState([]);
  const [isRecording, setIsRecording] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [popupMessage, setPopupMessage] = useState('');
  const [currentStep, setCurrentStep] = useState('idle');
  const [agentStatus, setAgentStatus] = useState('Ready');
  const [showInterface, setShowInterface] = useState(false);

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const streamRef = useRef(null);
  const sessionIdRef = useRef(null);

  useEffect(() => {
    const timer = setTimeout(() => setShowInterface(true), 500);
    return () => clearTimeout(timer);
  }, []);

  const showMessage = (msg) => {
    setPopupMessage(msg);
    setTimeout(() => setPopupMessage(''), 3000);
  };

  const playAudioResponse = async (audioBase64) => {
    try {
      const binary = atob(audioBase64);
      const buffer = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) buffer[i] = binary.charCodeAt(i);

      const audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const audioBuffer = await audioContext.decodeAudioData(buffer.buffer);
      const source = audioContext.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(audioContext.destination);

      return new Promise(resolve => {
        source.onended = resolve;
        source.start();
      });
    } catch (err) {
      console.error('Audio playback error:', err);
    }
  };

  const initializeAudio = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = e => audioChunksRef.current.push(e.data);
      mediaRecorderRef.current.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        audioChunksRef.current = [];
        await sendAudioToBackend(audioBlob);
      };
      return true;
    } catch (err) {
      showMessage('Microphone permission needed');
      return false;
    }
  };

  const sendAudioToBackend = async (audioBlob) => {
    setCurrentStep('processing');
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.webm');
    formData.append('session_id', sessionIdRef.current);
    try {
      const res = await fetch('http://localhost:5000/api/voice-chat', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      if (data.success) {
        setConversation(prev => [...prev, { type: 'user', text: data.user_text }, { type: 'agent', text: data.agent_response }]);
        setAgentStatus('Speaking...');
        await playAudioResponse(data.audio_response);
        setAgentStatus(data.status || 'Listening...');
        await startRecording();
      }
    } catch (err) {
      showMessage('Communication error');
    } finally {
      setCurrentStep('idle');
    }
  };

  const startRecording = async () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'inactive') {
      mediaRecorderRef.current.start();
      setIsRecording(true);
      setCurrentStep('listening');
      setAgentStatus('Listening...');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const startCall = async () => {
    if (!await initializeAudio()) return;
    const res = await fetch('http://localhost:5000/api/voice-init');
    const data = await res.json();
    if (data.success) {
      sessionIdRef.current = data.session_id;
      setConversation([{ type: 'agent', text: data.agent_response }]);
      await playAudioResponse(data.audio_response);
      await startRecording();
      setIsConnected(true);
    }
  };

  const handleMicClick = () => {
    if (!isConnected) startCall();
    else isRecording ? stopRecording() : startRecording();
  };

  const disconnect = () => {
    streamRef.current?.getTracks().forEach(t => t.stop());
    setIsConnected(false);
    setIsRecording(false);
    setConversation([]);
    setAgentStatus('Ready');
    sessionIdRef.current = null;
  };

  // Helper function to create animated motion divs
  const AnimatedBgDiv = ({ className, animate, transition, style }) => (
    <div 
      className={className}
      style={{
        ...style,
        animation: `floatAnimation ${transition?.duration || 8}s ease-in-out infinite`
      }}
    />
  );

  // Helper function for floating particles
  const FloatingParticle = ({ index }) => (
    <div
      key={index}
      className="absolute w-1 h-1 bg-white opacity-20 rounded-full"
      style={{
        left: `${Math.random() * 100}%`,
        top: `${Math.random() * 100}%`,
        animation: `particleFloat ${Math.random() * 20 + 10}s linear infinite`
      }}
    />
  );

  return (
    <>
      <style jsx>{`
        @keyframes floatAnimation {
          0%, 100% {
            transform: translate(0, 0) scale(1);
          }
          33% {
            transform: translate(100px, -50px) scale(1.2);
          }
          66% {
            transform: translate(-80px, 60px) scale(0.8);
          }
        }
        
        @keyframes particleFloat {
          0% {
            transform: translate(0, 0);
            opacity: 0.2;
          }
          50% {
            opacity: 0.4;
          }
          100% {
            transform: translate(${Math.random() * 200 - 100}px, ${Math.random() * 200 - 100}px);
            opacity: 0.1;
          }
        }
        
        @keyframes rotate {
          from {
            transform: rotate(0deg);
          }
          to {
            transform: rotate(360deg);
          }
        }
        
        @keyframes pulse {
          0%, 100% {
            transform: scale(1);
          }
          50% {
            transform: scale(1.2);
          }
        }
        
        .rotate {
          animation: rotate 2s linear infinite;
        }
        
        .pulse {
          animation: pulse 0.5s ease-in-out infinite;
        }
      `}</style>
      
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-black flex flex-col items-center justify-center relative overflow-hidden">
        {/* Animated Background Elements */}
        <div className="absolute inset-0 opacity-20">
          <AnimatedBgDiv 
            className="absolute top-1/4 left-1/4 w-64 h-64 bg-gradient-to-r from-green-400 to-blue-500 rounded-full blur-3xl"
            transition={{ duration: 8 }}
          />
          <AnimatedBgDiv 
            className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-gradient-to-r from-purple-400 to-pink-500 rounded-full blur-3xl"
            transition={{ duration: 10 }}
          />
        </div>

        {/* Floating Particles */}
        <div className="absolute inset-0 overflow-hidden">
          {[...Array(15)].map((_, i) => (
            <FloatingParticle key={i} index={i} />
          ))}
        </div>

        {/* Live Conversation Panel */}
        <div 
          className="absolute top-4 left-4 w-72 max-h-60 overflow-y-auto bg-black bg-opacity-70 border border-gray-700 rounded-xl p-4 text-white text-sm shadow-xl backdrop-blur-md z-50"
          style={{
            opacity: showInterface ? 1 : 0,
            transform: showInterface ? 'translateX(0)' : 'translateX(-100px)',
            transition: 'all 0.6s ease-out 0.5s'
          }}
        >
          <h2 className="font-semibold mb-2 text-yellow-400">Live Conversation</h2>
          <div className="space-y-2">
            {conversation.length === 0 ? (
              <p className="italic text-gray-400">No messages yet...</p>
            ) : (
              conversation.map((msg, idx) => (
                <div 
                  key={idx} 
                  className={`text-sm flex gap-2 ${msg.type === 'user' ? 'text-blue-400' : 'text-green-400'}`}
                  style={{
                    opacity: 0,
                    transform: 'translateY(10px)',
                    animation: `fadeInUp 0.3s ease-out ${idx * 0.1}s forwards`
                  }}
                >
                  <span className="font-semibold">
                    {msg.type === 'user' ? '🧑‍💬 You:' : '🤖 Agent Shreyash:'}
                  </span>
                  <span>{msg.text}</span>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Status Panel */}
        <div
          className="absolute top-4 right-4 bg-black bg-opacity-70 border border-gray-700 rounded-xl p-4 text-white text-sm shadow-xl backdrop-blur-md z-50"
          style={{
            opacity: showInterface ? 1 : 0,
            transform: showInterface ? 'translateX(0)' : 'translateX(100px)',
            transition: 'all 0.6s ease-out 0.7s'
          }}
        >
          <h2 className="font-semibold mb-2 text-cyan-400">Status</h2>
          <div className="flex items-center gap-2">
            <div 
              className={`w-3 h-3 rounded-full ${
                isConnected ? (isRecording ? 'bg-green-500' : 'bg-yellow-500') : 'bg-gray-500'
              } ${isRecording ? 'pulse' : ''}`}
            />
            <span className="text-white">{agentStatus}</span>
          </div>
          <div className="mt-2 text-xs text-gray-400">
            Step: {currentStep}
          </div>
        </div>

        {/* Main Header */}
        <div
          className="mb-12 z-10"
          style={{
            opacity: showInterface ? 1 : 0,
            transform: showInterface ? 'scale(1) translateY(-60px)' : 'scale(0.5) translateY(-50px)',
            transition: 'all 0.8s ease-out 0.2s'
          }}
        >
          <h1
            className="text-4xl md:text-5xl font-bold text-yellow-400 tracking-wide text-center"
            style={{
              opacity: showInterface ? 1 : 0,
              transform: showInterface ? 'scale(1)' : 'scale(0.3)',
              transition: 'all 1s ease-out 0.3s'
            }}
          >
            AI Project
          </h1>
        </div>

        {/* Main Voice Interface Circle */}
        <div
          className="relative w-80 h-80 mb-8 z-10"
          style={{
            opacity: showInterface ? 1 : 0,
            transform: showInterface ? 'scale(1)' : 'scale(0)',
            transition: 'all 0.6s ease-out 0.4s'
          }}
        >
          {/* Outer Rotating Ring */}
          <div
            className={`absolute inset-0 rounded-full bg-gradient-conic from-green-400 via-pink-500 via-purple-500 to-green-400 p-1 shadow-2xl shadow-purple-500 shadow-opacity-30 ${
              isConnected && isRecording ? 'rotate' : ''
            }`}
            style={{
              background: 'conic-gradient(from 0deg, #4ade80, #ec4899, #a855f7, #4ade80)'
            }}
          >
            <div className="w-full h-full rounded-full bg-gray-900"></div>
          </div>

          {/* Inner Circle with Controls */}
          <div className="absolute inset-4 rounded-full bg-gradient-to-br from-cyan-200 from-opacity-50 via-blue-300 via-opacity-50 to-indigo-400 to-opacity-50 backdrop-blur-sm shadow-xl shadow-cyan-400 shadow-opacity-20 border border-cyan-400 border-opacity-30 flex flex-col items-center justify-center gap-6">
            
            {/* Main Control Button */}
            <button
              onClick={handleMicClick}
              className="w-24 h-24 bg-black bg-opacity-80 backdrop-blur-sm rounded-full flex items-center justify-center text-white hover:bg-black hover:bg-opacity-90 transition-all duration-300 border border-white border-opacity-20 z-10 shadow-xl transform hover:scale-105 active:scale-95"
              disabled={currentStep === 'processing'}
            >
              <div
                className={isRecording ? 'pulse' : ''}
                style={{
                  animation: currentStep === 'processing' ? 'rotate 1s linear infinite' : ''
                }}
              >
                {currentStep === 'processing' ? (
                  <div className="w-8 h-8 border-2 border-white border-t-transparent rounded-full animate-spin" />
                ) : isRecording ? (
                  <MicOff className="w-8 h-8 text-red-400" />
                ) : (
                  <Mic className="w-8 h-8 text-white" />
                )}
              </div>
            </button>

            {/* Button Label */}
            <span 
              className="text-white text-sm font-medium text-center"
              style={{
                opacity: currentStep === 'processing' ? 0.5 : 1,
                transition: 'opacity 0.3s ease'
              }}
            >
              {currentStep === 'processing' ? 'Processing...' : 
               isRecording ? 'Stop Recording' : 
               isConnected ? 'Start Listening' : 'Start Call'}
            </span>

            {/* Disconnect Button */}
            {isConnected && (
              <button
                onClick={disconnect}
                className="flex items-center gap-2 px-4 py-2 bg-red-600 bg-opacity-80 hover:bg-red-600 text-white text-sm rounded-full transition-all duration-300 backdrop-blur-sm border border-red-400 border-opacity-30 transform hover:scale-105 active:scale-95"
                style={{
                  opacity: isConnected ? 1 : 0,
                  transform: isConnected ? 'scale(1)' : 'scale(0.8)',
                  transition: 'all 0.3s ease'
                }}
              >
                <PhoneOff className="w-4 h-4" />
                End Call
              </button>
            )}
          </div>
        </div>

        {/* Powered By */}
        <div
          className="text-gray-400 text-sm z-10"
          style={{
            opacity: showInterface ? 1 : 0,
            transform: showInterface ? 'translateY(0)' : 'translateY(20px)',
            transition: 'all 0.5s ease-out 0.8s'
          }}
        >
          Powered by <span className="text-white font-medium"></span>
        </div>

        {/* Enhanced Popup Messages */}
        {popupMessage && (
          <div
            className="fixed bottom-20 left-1/2 transform -translate-x-1/2 bg-black bg-opacity-90 backdrop-blur-sm border border-white border-opacity-20 rounded-lg px-6 py-3 text-white z-50 shadow-xl"
            style={{
              opacity: popupMessage ? 1 : 0,
              transform: popupMessage ? 'translate(-50%, 0) scale(1)' : 'translate(-50%, 50px) scale(0.8)',
              transition: 'all 0.3s ease'
            }}
          >
            <div className="flex items-center gap-3">
              <div className="w-3 h-3 rounded-full bg-yellow-500 pulse" />
              <span className="text-sm font-medium">{popupMessage}</span>
            </div>
          </div>
        )}

        <style jsx>{`
          @keyframes fadeInUp {
            to {
              opacity: 1;
              transform: translateY(0);
            }
          }
        `}</style>
      </div>
    </>
  );
};

export default VoiceAssistant;