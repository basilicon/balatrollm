<div align="center">
  <h1>BalatroLLM</h1>
  <p align="center">
    <a href="https://github.com/coder/balatrollm/releases">
      <img alt="GitHub release" src="https://img.shields.io/github/v/release/coder/balatrollm?include_prereleases&sort=semver&style=for-the-badge&logo=github"/>
    </a>
    <a href="https://discord.gg/TPn6FYgGPv">
      <img alt="Discord" src="https://img.shields.io/badge/discord-server?style=for-the-badge&logo=discord&logoColor=%23FFFFFF&color=%235865F2"/>
    </a>
  </p>
  <div><img src="./docs/assets/balatrollm.svg" alt="balatrobot" width="170" height="170"></div>
  <p><em>A Balatro bot powered by LLMs</em></p>
</div>

---

BalatroLLM is a bot that uses Large Language Models (LLMs) to play [Balatro](https://www.playbalatro.com/), the popular roguelike poker deck-building game. The bot analyzes game states, makes strategic decisions, and executes actions through the [BalatroBot](https://github.com/coder/balatrobot) API.

## ✨ Custom Enhancements

This repository features several major improvements over the original framework:

- **Genetic Algorithm Optimization (`deterministic_player.py`)**: The deterministic player now incorporates a powerful `optimize_strategy` feature. It uses a Genetic Algorithm to rapidly simulate thousands of possible game actions, dynamically finding the most mathematically optimal target hand weights and discard aggressiveness.
- **Enhanced Default Strategy (`strategies/default`)**: We've introduced a highly modular and robust prompt strategy system utilizing Jinja templating (`GAMESTATE.md.jinja`, `MEMORY.md.jinja`, `STRATEGY.md.jinja`). This structure gives the LLM clearer context about the current game state, available actions, and memory, driving vastly superior decision-making.

## 📚 Documentation

https://coder.github.io/balatrollm/

## 🚀 Related Projects

- [**BalatroBot**](https://github.com/coder/balatrobot): API for developing Balatro bots
- [**BalatroLLM**](https://github.com/coder/balatrollm): Play Balatro with LLMs
- [**BalatroBench**](https://github.com/coder/balatrobench): Benchmark LLMs playing Balatro

<br>

<h3 align="center">
  BalatroBot + BalatroLLM = <a href="https://balatrobench.com/">BalatroBench.com</a>
</h3>

<br>

<img width="3338" height="2649" alt="Main BalatroBench" src="https://github.com/user-attachments/assets/9cb34581-0717-4989-a654-7f378a5e02a5" />

<br>

<img width="3362" height="2649" alt="Run Viewer BalatroBench" src="https://github.com/user-attachments/assets/2cfe97ad-11db-43f2-b25a-296e4d86a12d" />

<br>

<img width="3338" height="2649" alt="Community BalatroBench" src="https://github.com/user-attachments/assets/a42549e8-ea4b-4100-8e45-2b39e1c53c15" />
