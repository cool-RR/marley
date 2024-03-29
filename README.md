# Marley #

Marley is a framework for multi-agent reinforcement learning. It lets you design a game and run experiments with multiple AI-powered agents playing that game.

**Marley isn't ready for use yet; it works for my research purposes, but I haven't yet made the work needed to make it easy to understand for others.** If you're still interested in using it, feel free to read the code or [email me](mailto:ram@rachum.com) for help.

I'm using Marley in my research mission to use machine learning to understand human society. I want to produce emergent reciprocation in selfish RL agents and use that to show social behavior. [Talk video](https://youtu.be/WMVqYYarFkQ), [slides](https://docs.google.com/presentation/d/1uBV2VkCfds_K3Aj7JimK1aE22AJSmLVoOg5oR4pOJU4/), [sign up for updates](https://groups.google.com/g/ram-rachum-research-announce).

I hope to make Marley easier to use soon.

## GridRoyale ##

![](https://i.imgur.com/pmxEKnR.gif)

## [View the live version here!](https://grid-royale.herokuapp.com/) ##

**GridRoyale** is a life simulation. It's a case study that's bundled with Marley.

GridRoyale is similar to [Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life) or
[GridWorld](https://towardsdatascience.com/reinforcement-learning-implement-grid-world-from-scratch-c5963765ebff),
except I added game mechanics to encourage the players to behave socially. These game mechanics are
similar to those in the [battle royale](https://en.wikipedia.org/wiki/Battle_royale_game) genre of
computer games, which is why it's called GridRoyale.

Players get 10 points for eating food, i.e. the dots that regenerate randomly. Players lose 5 points for colliding with other agents or walls. Players lose 10 points for being shot.


# How to run GridRoyale #

Installation:

```console
$ pip install marley
```

Run the server:

```console
$ marley grid_royale demo
```

This will automatically open a browser window and show you your simulation.
