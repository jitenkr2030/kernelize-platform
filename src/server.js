#!/usr/bin/env node

/**
 * KERNELIZE Backend Server
 * Advanced Data Management & Integration Platform
 * 
 * This server provides comprehensive data pipeline processing,
 * cloud integration, serverless deployment, and CDN management.
 */

const app = require('./dist/app');

// Start the server
const PORT = process.env.PORT || 8000;
const HOST = process.env.HOST || '0.0.0.0';

app.listen(PORT, HOST, () => {
  console.log(`
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║          KERNELIZE Backend API Server                          ║
║          Advanced Data Management & Integration                ║
║                                                                ║
║  🌐 Server:     http://${HOST}:${PORT}                            ║
║  📊 Health:     http://${HOST}:${PORT}/health                      ║
║  🔌 WebSocket:  ws://${HOST}:${PORT}                               ║
║  📖 API Docs:   http://${HOST}:${PORT}/api/v1                       ║
║                                                                ║
║  🚀 Features:                                                ║
║     • ETL Pipeline Processing                                ║
║     • Data Validation & Quality                              ║
║     • Schema Management                                      ║
║     • Cloud Storage Integration                              ║
║     • Serverless Function Deployment                         ║
║     • CDN Management                                         ║
║     • Real-time Monitoring                                   ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
  `);
});

// Graceful shutdown handling
process.on('SIGTERM', () => {
  console.log('Received SIGTERM, shutting down gracefully...');
  process.exit(0);
});

process.on('SIGINT', () => {
  console.log('Received SIGINT, shutting down gracefully...');
  process.exit(0);
});

process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
});

process.on('uncaughtException', (error) => {
  console.error('Uncaught Exception:', error);
  process.exit(1);
});