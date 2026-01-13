# KERNELIZE Platform Frontend

A modern, responsive web dashboard for the KERNELIZE advanced AI-powered knowledge compression platform.

## Features

- 🎯 **Real-time Dashboard**: Live monitoring of compression jobs, system metrics, and user activity
- 🔍 **Interactive Query Interface**: Advanced compression query builder with streaming support
- 📊 **Analytics & Visualizations**: Comprehensive performance analytics with interactive charts
- 🌐 **Knowledge Graph**: D3.js-powered interactive visualization of compressed knowledge relationships
- 👥 **User Management**: Role-based access control and user administration
- ⚙️ **System Monitoring**: Real-time system health and performance monitoring
- 🎨 **Modern UI**: Material-UI components with dark theme and responsive design
- 🔐 **Secure Authentication**: JWT-based authentication with automatic token refresh

## Technology Stack

- **React 18** with TypeScript
- **Material-UI (MUI)** for components and theming
- **Vite** for fast development and building
- **React Query** for server state management
- **Apollo Client** for GraphQL operations
- **Zustand** for client state management
- **Recharts** for data visualization
- **D3.js** for advanced visualizations
- **React Hook Form** for form handling

## Quick Start

### Prerequisites

- Node.js 18+ 
- npm or yarn

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd kernelize/frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The application will be available at `http://localhost:3000`

### Development Scripts

```bash
npm run dev          # Start development server
npm run build        # Build for production
npm run preview      # Preview production build
npm run lint         # Run ESLint
npm run type-check   # TypeScript type checking
```

## Configuration

### Environment Variables

Create a `.env` file in the frontend directory:

```env
VITE_API_BASE_URL=http://localhost:8000/api
VITE_GRAPHQL_URL=http://localhost:8000/graphql
VITE_WS_URL=ws://localhost:8000/ws
```

### API Integration

The frontend is configured to work with the KERNELIZE backend API. Make sure the backend is running and accessible at the configured URLs.

### Development Setup

1. **Backend Integration**: Ensure the KERNELIZE backend is running with CORS enabled for `http://localhost:3000`
2. **Database**: Set up the database and run migrations
3. **Authentication**: Configure JWT secret and authentication endpoints

## Project Structure

```
frontend/
├── src/
│   ├── components/          # Reusable UI components
│   │   ├── Auth/           # Authentication components
│   │   └── Layout/         # Layout components
│   ├── pages/              # Page components
│   │   ├── Analytics/      # Analytics dashboard
│   │   ├── Auth/           # Login/auth pages
│   │   ├── CompressionJobs/# Job management
│   │   ├── Dashboard/      # Main dashboard
│   │   ├── KnowledgeGraph/ # Knowledge visualization
│   │   ├── QueryInterface/ # Query builder
│   │   ├── Settings/       # User settings
│   │   ├── SystemMonitor/  # System monitoring
│   │   └── UserManagement/ # User administration
│   ├── store/              # State management
│   ├── App.tsx             # Main application
│   ├── main.tsx            # Application entry point
│   ├── theme.ts            # Material-UI theme
│   └── index.css           # Global styles
├── public/                 # Static assets
├── package.json
├── vite.config.ts
├── tsconfig.json
└── index.html
```

## Key Components

### Dashboard
Real-time overview of system status, compression metrics, and user activity.

### Query Interface
Interactive tool for building and executing compression queries with advanced options.

### Analytics
Comprehensive analytics with interactive charts for performance monitoring.

### Knowledge Graph
D3.js-powered visualization of knowledge relationships and concepts.

### System Monitor
Real-time system health monitoring with performance metrics.

## State Management

- **Zustand**: Client-side state (authentication, UI state)
- **React Query**: Server state (API data, caching, synchronization)
- **Apollo Client**: GraphQL state management

## API Integration

### REST API
- User authentication and profile management
- Compression job management
- Analytics and reporting
- System monitoring

### GraphQL
- Complex queries and mutations
- Real-time subscriptions
- Efficient data fetching

### WebSocket
- Real-time updates for monitoring
- Live compression job progress
- System alerts and notifications

## Security

- JWT token-based authentication
- Automatic token refresh
- Role-based access control
- CSRF protection
- XSS prevention

## Performance

- Code splitting and lazy loading
- Virtual scrolling for large datasets
- Optimistic updates
- Image optimization
- Service worker caching

## Responsive Design

- Mobile-first approach
- Adaptive layouts
- Touch-friendly interfaces
- Progressive Web App features

## Accessibility

- ARIA labels and roles
- Keyboard navigation
- Screen reader support
- High contrast mode
- WCAG 2.1 compliance

## Testing

```bash
npm run test           # Run unit tests
npm run test:e2e       # Run end-to-end tests
npm run test:coverage  # Generate coverage report
```

## Building for Production

```bash
# Build optimized production bundle
npm run build

# Preview production build locally
npm run preview

# Deploy to your hosting platform
# The dist/ folder contains the built application
```

## Deployment

### Docker

```dockerfile
FROM node:18-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

### Static Hosting

The built application can be deployed to any static hosting service:

- Vercel
- Netlify
- AWS S3 + CloudFront
- Azure Static Web Apps
- GitHub Pages

## Development Guidelines

### Code Style
- ESLint configuration for consistent code style
- Prettier for code formatting
- TypeScript for type safety
- Conventional commits

### Component Development
- Functional components with hooks
- PropTypes or TypeScript interfaces
- Consistent naming conventions
- Reusable component patterns

### State Management
- Local state for component-specific data
- Global state for shared data
- Server state via React Query
- Proper cleanup and unsubscriptions

## Troubleshooting

### Common Issues

1. **CORS Errors**: Ensure backend has CORS enabled for localhost:3000
2. **API Connection**: Check API URLs in environment variables
3. **Authentication**: Verify JWT secret and token expiration
4. **Build Errors**: Clear node_modules and reinstall dependencies

### Development Tips

1. Use React Developer Tools for debugging
2. Enable network logging in browser DevTools
3. Check console for error messages
4. Use TypeScript strict mode for better type safety

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new features
5. Run linting and type checking
6. Submit a pull request

## License

This project is part of the KERNELIZE Platform. See the main project license for details.

## Support

For support and questions:
- Check the documentation
- Review existing issues
- Create a new issue with detailed information
- Contact the development team

---

Built with ❤️ using React, TypeScript, and Material-UI