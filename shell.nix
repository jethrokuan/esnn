with import <nixpkgs> {};
let
  pythonEnv = poetry2nix.mkPoetryEnv {
    python = python3;
    poetrylock = ./poetry.lock;
    overrides = [
      pkgs.poetry2nix.defaultPoetryOverrides
    ];
  };
in
pkgs.mkShell {
  name = "esnn";
  nativeBuildInputs = [
    pythonEnv
    poetry
  ];
}
