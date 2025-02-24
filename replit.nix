{pkgs}: {
  deps = [
    pkgs.libGLU
    pkgs.libGL
    pkgs.xsimd
    pkgs.pkg-config
    pkgs.libxcrypt
    pkgs.portaudio
    pkgs.ffmpeg-full
    pkgs.glibcLocales
  ];
}
