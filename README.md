#

- `cargo build --release -p asr`

- `cargo run --release -p asr transcribe --audio "path/to/audio.wav"`

- `.\target\release\asr.exe transcribe --audio "audio_files/file_name.wav"`

- `.\target\release\asr.exe openai --ip 127.0.0.1 --port 3001`

### Build on MacOS with Github Actions
- ``` sh
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/`env"

    brew install libsoxr pkg-config

    cd $GITHUB_WORKSPACE

    cargo build --release --features coreml -p asr

    mkdir -p dist
    cp target/release/asr dist/
    cp /opt/homebrew/lib/libsoxr.dylib dist/

    tar -czvf asr-macos-arm64.tar.gz -C dist .

    gh release create v1.0.0 asr-macos-arm64.tar.gz \
    --title "ASR macOS ARM64" \
    --notes "macOS Apple Silicon build with CoreML support"
```