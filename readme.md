# AI Content Blocker Chrome Extension (Manifest V3)

## Overview

This Chrome extension detects and optionally blocks AI-generated or AI-assisted content on web pages. It works in conjunction with a local backend server (`server.py`) that performs text analysis and returns classification results.

The extension is designed for Manifest V3 and follows modern Chrome extension architecture using a service worker background script and message-based communication between components.

---

## Features

- Detects AI-generated or AI-assisted text on web pages
- Highlights or blocks flagged content based on confidence threshold
- Optional removal of Google AI Overview sections
- Real-time scanning of dynamically loaded content via MutationObserver
- Local backend integration for classification (Python server)
- Settings panel with:
  - Enable/disable extension
  - Toggle Google AI Overview removal
  - Confidence threshold control
  - Live statistics (scanned vs blocked content)
  - Server health status indicator

---

## Architecture

### Manifest V3 Design

- **content.js**
  - Injected into all pages
  - Extracts readable text blocks
  - Sends text to background service worker
  - Applies UI changes (dim, block, or reveal content)

- **background.js (Service Worker)**
  - Handles all network communication with local server
  - Routes messages between content scripts and storage
  - Manages:
    - `/detect` requests for AI classification
    - `/health` checks for server status
    - Settings persistence via `chrome.storage.local`

- **server.py (Local Backend)**
  - Processes text classification requests
  - Returns AI detection results and confidence scores
  - Must be running locally for extension functionality

- **popup.html / popup.js**
  - User interface for extension control
  - Displays status, statistics, and settings
  - Allows runtime configuration changes

---

## Installation

### 1. Extract Extension
Unzip the provided extension package into a local folder.

### 2. Load into Chrome
1. Open `chrome://extensions`
2. Enable **Developer mode** (top right corner)
3. Click **Load unpacked**
4. Select the extracted extension folder

### 3. Start Backend Server
Ensure the Python backend is running:

```bash
python server.py
