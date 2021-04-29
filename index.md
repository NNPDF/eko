---
layout: default
title: Home
---

# Evolutionary Kernel Operators

## Get Started

Setup the development environment:

- add the missing tools
  - ruby headers (required for some packages)
  - `bundler` for environment isolation
- install the local environment & run locally

```bash
sudo apt install ruby-dev
sudo gem install bundler
# in the repo root
bundle config set --local path 'vendor/bundle'
bundle install
bundle exec jekyll serve
```

## Syntax reference

Actually it's just HTML/Markdown, but to have an example on how it is rendered
look at [this page](example-content).
