# Eko website

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
