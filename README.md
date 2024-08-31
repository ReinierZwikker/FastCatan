# FastCatan

This repository contains an implementation of 'The Settlers of Catan',
suitable for development of Neural Nets (NN) and other algorithmic player to play the game.
It consists of a Catan Engine that can play catan games,
an extensive GUI that can be used to set up, monitor, and play games of Catan,
and can simultaneously be used to set up and monitor training sessions.

## Compilation

To compile the code, the user needs to have the NVIDIA user toolkit available at the following link:
https://developer.nvidia.com/cuda-toolkit. Additionally, the user should also use the Microsoft Windows OS and have OpenGL and SDL2 installed.

To run the code, the user needs an NVIDIA graphics card on a Windows system.

## Structure

There are four branches in this repository. This is the `BeanAI` branch, which contains the NN made by Mauro Beenders, and does require CUDA.

### NN Code structure

The Neural Net implementation and its connection to Catan is in `./src/game/AIPlayer/ai_bean_player.cpp/h`. 

The files `./src/game/AIHelpers/bean_helper.cpp/h` handle training.

### Game Structure

A game consists of 4 players, that each play a turn to complete a round, and play rounds until one player wins or the round limit is reached.

## Players

Various players have been written for FastCatan:

1. A Console Player, to play from the command line
2. A Gui Player, to play the basic moves from a GUI
3. A Random Player, this player selects a random available move to play
4. A ZwikAI Player, the NSSR Neural Net created by Reinier Zwikker
5. A BeanAI Player, the Neural Net created by Mauro Beenders
