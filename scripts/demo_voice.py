# scripts/demo_voice.py
"""
Voice-Based Test Generation Demo - Production Ready

Features:
✅ Automatic mode selection (local FREE or cloud PAID)
✅ Clear cost visibility
✅ Multiple demo modes (single, continuous, interactive)
✅ Backend switching
✅ Comprehensive metrics

Usage:
    # Quick start (auto-mode)
    python scripts/demo_voice.py

    # Single command mode
    python scripts/demo_voice.py --single

    # Continuous listening
    python scripts/demo_voice.py --continuous

    # Force local mode (FREE)
    python scripts/demo_voice.py --mode local

    # Force cloud mode (PAID)
    python scripts/demo_voice.py --mode cloud
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import voice interface
from modules.voice_interface import VoiceInterface, VoiceConfig, VoiceEnabledAgent, WhisperMode


# ==================== Banner ====================

def print_banner(config: VoiceConfig, backend_info: dict):
    """Print welcome banner with configuration info"""
    print("\n" + "="*70)
    print("🎤 AI QA Agent - Voice Interface Demo")
    print("="*70)
    print(f"\n📊 Active Configuration:")
    print(f"   Mode: {config.mode.value}")
    print(f"   Backend: {backend_info['name']}")
    print(f"   Cost: ${backend_info['cost_per_minute']}/minute", end="")
    
    if backend_info['cost_per_minute'] == 0:
        print(" ✅ FREE")
    else:
        print(" 💰 PAID")
    
    print(f"   Internet: {'Required' if backend_info['requires_internet'] else 'Not required'}")
    print(f"   Language: {config.language}")
    print(f"   TTS: {'Enabled' if config.enable_tts else 'Disabled'}")
    print(f"   Wake Words: {', '.join(config.wake_words)}")
    print("="*70 + "\n")


# ==================== Check Dependencies ====================

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...\n")
    
    all_ok = True
    
    # PyAudio (required)
    try:
        import pyaudio
        print("   ✅ PyAudio installed")
    except ImportError:
        print("   ❌ PyAudio not installed")
        print("      Install: pip install pyaudio")
        all_ok = False
    
    # Whisper local (optional)
    try:
        import whisper
        print("   ✅ Whisper (local) available")
    except ImportError:
        print("   ⚠️  Whisper (local) not installed")
        print("      Install: pip install openai-whisper")
    
    # Faster-whisper (optional, but recommended)
    try:
        from faster_whisper import WhisperModel
        print("   ✅ Faster-Whisper available (5-10x faster!)")
    except ImportError:
        print("   ⚠️  Faster-Whisper not installed (recommended)")
        print("      Install: pip install faster-whisper")
    
    # OpenAI API (optional)
    try:
        from openai import OpenAI
        if os.getenv("OPENAI_API_KEY"):
            print("   ✅ OpenAI API configured (cloud mode available)")
        else:
            print("   ⚠️  OPENAI_API_KEY not set (cloud mode unavailable)")
    except ImportError:
        print("   ⚠️  OpenAI SDK not installed")
        print("      Install: pip install openai")
    
    # TTS (optional)
    try:
        from gtts import gTTS
        import pygame
        print("   ✅ TTS (gTTS + pygame) available")
    except ImportError:
        print("   ⚠️  TTS not available")
        print("      Install: pip install gtts pygame")
    
    print()
    
    if not all_ok:
        print("❌ Missing required dependencies. Please install them first.\n")
        sys.exit(1)


# ==================== Demo Modes ====================

def demo_single_command(config: VoiceConfig):
    """
    Single command mode: Record once, process, exit
    """
    print("\n" + "="*70)
    print("🎙️ Single Command Mode")
    print("="*70)
    print("\nThis will:")
    print("  1. Record your voice for 10 seconds")
    print("  2. Transcribe using configured backend")
    print("  3. Execute the command")
    print("  4. Exit")
    print("="*70 + "\n")
    
    # Create voice interface (no continuous listening)
    voice = VoiceInterface(config=config)
    backend_info = voice.get_backend_info()
    
    print(f"Backend: {backend_info['name']} (${backend_info['cost_per_minute']}/min)\n")
    print("🔴 Recording will start in 3 seconds...")
    print("Say: 'Test [URL]' or 'Check [URL]'\n")
    
    import time
    for i in range(3, 0, -1):
        print(f"   {i}...")
        time.sleep(1)
    
    print("\n🎙️ RECORDING NOW! Speak clearly...\n")
    
    # Record audio
    import pyaudio
    import wave
    
    try:
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1024
        )
        
        frames = []
        for _ in range(0, int(16000 / 1024 * 10)):  # 10 seconds
            data = stream.read(1024, exception_on_overflow=False)
            frames.append(data)
        
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        print("✅ Recording complete!\n")
        
        # Save audio
        temp_file = Path("temp_voice_command.wav")
        with wave.open(str(temp_file), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(16000)
            wf.writeframes(b''.join(frames))
        
        # Transcribe
        print("🎧 Transcribing...")
        text = voice._transcribe(temp_file)
        
        print(f"\n📝 You said: '{text}'\n")
        print("="*70)
        
        temp_file.unlink(missing_ok=True)
        
        if not text:
            print("\n❌ No speech detected. Please try again.")
            return
        
        # Execute command
        execute_voice_command(text, voice)
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
    
    # Show metrics
    metrics = voice.get_metrics()
    print(f"\n📊 Metrics:")
    print(json.dumps(metrics, indent=2))


def demo_continuous(config: VoiceConfig):
    """
    Continuous mode: Keep listening until stopped
    """
    print("\n" + "="*70)
    print("🎤 Continuous Listening Mode")
    print("="*70)
    print("\nThis will:")
    print("  1. Continuously listen for voice commands")
    print("  2. Activate on wake word detection")
    print("  3. Execute commands automatically")
    print("  4. Run until 'stop' or Ctrl+C")
    print("="*70 + "\n")
    
    # Create voice-enabled agent
    from modules.conversational_agent import ConversationalAgent
    
    agent = VoiceEnabledAgent(
        agent=ConversationalAgent(),
        config=config
    )
    
    backend_info = agent.voice.get_backend_info()
    
    print(f"Backend: {backend_info['name']}")
    print(f"Cost: ${backend_info['cost_per_minute']}/minute\n")
    print(f"Wake words: {', '.join(config.wake_words)}")
    print("Example: 'Test https://jsonplaceholder.typicode.com'\n")
    print("Say 'stop' or press Ctrl+C to quit\n")
    print("="*70 + "\n")
    
    try:
        agent.start()
        input("Press Enter to stop listening...\n")
    
    except KeyboardInterrupt:
        print("\n\n⏹️ Interrupted by user")
    
    finally:
        agent.stop()
        
        # Show metrics
        metrics = agent.get_metrics()
        print(f"\n📊 Session Metrics:")
        print(json.dumps(metrics, indent=2))


def demo_interactive(config: VoiceConfig):
    """
    Interactive mode: Menu-driven interface
    """
    from modules.conversational_agent import ConversationalAgent
    
    agent = VoiceEnabledAgent(
        agent=ConversationalAgent(),
        config=config
    )
    
    while True:
        print("\n" + "="*70)
        print("📋 Interactive Menu")
        print("="*70)
        print("  1. Start continuous listening")
        print("  2. Record single command")
        print("  3. View current configuration")
        print("  4. View metrics")
        print("  5. View command history")
        print("  6. Switch backend (local ↔ cloud)")
        print("  7. Test with sample URL")
        print("  8. Exit")
        print("="*70)
        
        choice = input("\nYour choice [1-8]: ").strip()
        
        if choice == "1":
            # Continuous listening
            try:
                backend_info = agent.voice.get_backend_info()
                print(f"\n🎤 Starting continuous listening...")
                print(f"Backend: {backend_info['name']} (${backend_info['cost_per_minute']}/min)")
                print("Say 'stop' when done.\n")
                
                agent.start()
                input("Press Enter to stop...\n")
                agent.stop()
            
            except KeyboardInterrupt:
                print("\n⏹️ Stopped")
                agent.stop()
        
        elif choice == "2":
            # Single command
            print("\n⚠️ Single command mode - use main menu option 1 instead")
        
        elif choice == "3":
            # Configuration
            backend_info = agent.voice.get_backend_info()
            print(f"\n📊 Current Configuration:")
            print(f"   Mode: {agent.voice.config.mode.value}")
            print(f"   Backend: {backend_info['name']}")
            print(f"   Model: {agent.voice.config.model_size}")
            print(f"   Language: {agent.voice.config.language}")
            print(f"   Cost: ${backend_info['cost_per_minute']}/min")
            print(f"   TTS: {'Enabled' if agent.voice.config.enable_tts else 'Disabled'}")
        
        elif choice == "4":
            # Metrics
            metrics = agent.get_metrics()
            print("\n📊 Current Metrics:")
            print(json.dumps(metrics, indent=2))
        
        elif choice == "5":
            # History
            history = agent.voice.get_command_history()
            print("\n📜 Recent Commands:")
            if history:
                for i, cmd in enumerate(history, 1):
                    print(f"  {i}. [{cmd['timestamp']}] {cmd['text']}")
            else:
                print("  No commands yet.")
        
        elif choice == "6":
            # Switch backend
            current_mode = agent.voice.config.mode.value
            print(f"\n🔄 Current mode: {current_mode}")
            print("Available modes:")
            print("  1. local (FREE)")
            print("  2. cloud (PAID)")
            print("  3. auto (fallback)")
            
            new_mode = input("Select mode [1-3]: ").strip()
            mode_map = {"1": "local", "2": "cloud", "3": "auto"}
            
            if new_mode in mode_map:
                agent.voice.config.mode = WhisperMode[mode_map[new_mode].upper()]
                agent.voice._init_whisper()
                print(f"✅ Switched to {mode_map[new_mode]} mode")
            else:
                print("❌ Invalid choice")
        
        elif choice == "7":
            # Test sample URL
            sample_url = "https://jsonplaceholder.typicode.com"
            print(f"\n🧪 Testing with: {sample_url}")
            
            sample_command = f"Test {sample_url}"
            agent.handle_voice_command(sample_command)
        
        elif choice == "8":
            print("\n👋 Goodbye!")
            break
        
        else:
            print("\n❌ Invalid choice. Please try again.")


# ==================== Command Execution ====================

def execute_voice_command(text: str, voice: VoiceInterface):
    """Execute a voice command"""
    print(f"💬 Executing: {text}\n")
    
    # Extract URL
    url = voice.validator.extract_url(text)
    
    if not url:
        print("❌ No URL detected in command")
        print("   Example: 'Test https://example.com'\n")
        voice.speak("Please provide a URL to test")
        return
    
    print(f"🎯 Target URL: {url}\n")
    voice.speak(f"Starting tests for {url}")
    
    try:
        from modules.conversational_agent import ConversationalAgent
        from modules.async_scraper import AsyncScraper
        from modules.test_generator import TestGenerator
        from modules.runner import Runner
        
        # Step 1: Parse
        print("Step 1: Parsing requirement...")
        agent = ConversationalAgent()
        req = agent.parse_requirement(text)
        print(f"✅ Intent: {req['intent']}\n")
        
        # Step 2: Scan
        print("Step 2: Scanning website...")
        scraper = AsyncScraper()
        scan_result = scraper.deep_scan(url)
        print(f"✅ Scanned {len(scan_result.get('pages', []))} pages\n")
        
        # Step 3: Generate
        print("Step 3: Generating test plan...")
        generator = TestGenerator()
        plan = generator.generate_plan(req, scan_result)
        
        # Fix URLs
        for suite in plan['suites'].get('api', []):
            suite['base_url'] = url
        for suite in plan['suites'].get('performance', []):
            suite['tool'] = 'lighthouse'
            suite['url'] = url
        
        total = sum(len(suites) for suites in plan['suites'].values())
        print(f"✅ Generated {total} test suites\n")
        
        # Step 4: Execute
        print("Step 4: Executing tests...")
        runner = Runner()
        results = runner.run(plan)
        
        # Results
        print("\n" + "="*70)
        print("✅ Execution Complete!")
        print("="*70)
        print(f"\n📊 Results:")
        print(f"   Run ID: {results['run_id']}")
        print(f"   Success: {results['success']}")
        print(f"   Duration: {results['metrics']['total_duration_s']:.2f}s")
        print(f"   Report: reports/{results['run_id']}.html\n")
        
        # Voice feedback
        if results['success']:
            voice.speak("Tests completed successfully!")
        else:
            voice.speak("Tests completed with some failures. Check the report.")
    
    except Exception as e:
        logger.error(f"Test execution failed: {e}", exc_info=True)
        print(f"\n❌ Error: {e}\n")
        voice.speak("Sorry, test execution failed.")


# ==================== Main ====================

def main():
    """Main entry point"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Voice-based test generation demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto mode (default)
  python scripts/demo_voice.py

  # Force local mode (FREE)
  python scripts/demo_voice.py --mode local

  # Force cloud mode (PAID, requires OPENAI_API_KEY)
  python scripts/demo_voice.py --mode cloud

  # Single command
  python scripts/demo_voice.py --single

  # Continuous listening
  python scripts/demo_voice.py --continuous
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["local", "cloud", "auto"],
        default="auto",
        help="Voice backend mode (default: auto)"
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Single command mode"
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Continuous listening mode"
    )
    parser.add_argument(
        "--model",
        default="base",
        help="Whisper model size (default: base)"
    )
    parser.add_argument(
        "--tts",
        action="store_true",
        help="Enable text-to-speech"
    )
    parser.add_argument(
        "--no-deps-check",
        action="store_true",
        help="Skip dependency check"
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    if not args.no_deps_check:
        check_dependencies()
    
    # Create config
    mode = WhisperMode[args.mode.upper()]
    
    config = VoiceConfig(
        mode=mode,
        model_size=args.model,
        language="en",
        enable_tts=args.tts,
        enable_metrics=True,
        enable_command_validation=True,
        use_faster_whisper=True
    )
    
    # Create voice interface to get backend info
    voice = VoiceInterface(config=config)
    backend_info = voice.get_backend_info()
    voice.stop_listening()
    
    # Print banner
    print_banner(config, backend_info)
    
    # Determine demo mode
    if args.single:
        demo_mode = "single"
    elif args.continuous:
        demo_mode = "continuous"
    else:
        demo_mode = "interactive"
    
    # Run selected mode
    try:
        if demo_mode == "single":
            demo_single_command(config)
        elif demo_mode == "continuous":
            demo_continuous(config)
        else:
            demo_interactive(config)
    
    except KeyboardInterrupt:
        print("\n\n⏹️ Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
