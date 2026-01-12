.PHONY: release-patch release-minor release-major release check-clean

# Get current version from latest tag, default to 0.0.0
CURRENT_VERSION := $(shell git describe --tags --abbrev=0 2>/dev/null | sed 's/^v//' || echo "0.0.0")
MAJOR := $(shell echo $(CURRENT_VERSION) | cut -d. -f1)
MINOR := $(shell echo $(CURRENT_VERSION) | cut -d. -f2)
PATCH := $(shell echo $(CURRENT_VERSION) | cut -d. -f3)

check-clean: ## Verify git is clean and synced with remote
	@if [ -n "$$(git status --porcelain)" ]; then \
		echo "Error: Working directory not clean. Commit or stash changes first."; \
		exit 1; \
	fi
	@git fetch origin
	@if [ "$$(git rev-parse HEAD)" != "$$(git rev-parse @{u})" ]; then \
		echo "Error: Local branch not synced with remote. Push or pull first."; \
		exit 1; \
	fi
	@echo "Git state OK: clean and synced"

release-patch: check-clean ## Release patch version (0.1.0 -> 0.1.1)
	@NEW_VERSION="$(MAJOR).$(MINOR).$$(($(PATCH)+1))" && \
	echo "Releasing v$$NEW_VERSION" && \
	git tag -a "v$$NEW_VERSION" -m "Release v$$NEW_VERSION" && \
	git push --tags

release-minor: check-clean ## Release minor version (0.1.0 -> 0.2.0)
	@NEW_VERSION="$(MAJOR).$$(($(MINOR)+1)).0" && \
	echo "Releasing v$$NEW_VERSION" && \
	git tag -a "v$$NEW_VERSION" -m "Release v$$NEW_VERSION" && \
	git push --tags

release-major: check-clean ## Release major version (0.1.0 -> 1.0.0)
	@NEW_VERSION="$$(($(MAJOR)+1)).0.0" && \
	echo "Releasing v$$NEW_VERSION" && \
	git tag -a "v$$NEW_VERSION" -m "Release v$$NEW_VERSION" && \
	git push --tags

release: release-patch ## Default: patch release
