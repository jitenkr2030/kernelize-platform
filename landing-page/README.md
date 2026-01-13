# KERNELIZE Platform Landing Page

Modern, high-converting landing page for the KERNELIZE Enterprise Platform built with Next.js 14, TypeScript, and Tailwind CSS.

## 🚀 Features

- **Modern Design**: Dark theme with glassmorphism effects and smooth animations
- **Responsive**: Fully responsive design for all screen sizes
- **Performance Optimized**: Built with Next.js 14 for optimal performance
- **SEO Ready**: Complete metadata and Open Graph tags
- **Animations**: Smooth scroll animations with Framer Motion
- **Type Safe**: Full TypeScript implementation

## 📁 Project Structure

```
landing-page/
├── app/
│   ├── globals.css        # Global styles and Tailwind directives
│   ├── layout.tsx         # Root layout with fonts and metadata
│   └── page.tsx           # Main landing page composition
├── components/
│   ├── layout/
│   │   ├── Footer.tsx     # Site footer
│   │   └── Navbar.tsx     # Sticky navigation header
│   ├── sections/
│   │   ├── CTA.tsx        # Call-to-action section
│   │   ├── Features.tsx   # Features grid section
│   │   ├── Hero.tsx       # Hero section with animations
│   │   ├── Pricing.tsx    # Pricing plans section
│   │   ├── Stats.tsx      # Statistics strip
│   │   └── Testimonials.tsx # Customer testimonials
│   └── ui/
│       ├── Button.tsx     # Reusable button component
│       └── Card.tsx       # Card component with variants
├── lib/
│   ├── constants.ts       # Site content and configuration
│   └── utils.ts           # Utility functions
├── package.json           # Dependencies
├── tailwind.config.ts     # Tailwind configuration
└── tsconfig.json          # TypeScript configuration
```

## 🛠️ Tech Stack

- **Framework**: Next.js 14 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Animations**: Framer Motion
- **Icons**: Lucide React
- **Utilities**: clsx, tailwind-merge

## 🚦 Getting Started

### Prerequisites

- Node.js 18+
- npm or yarn

### Installation

```bash
# Navigate to landing page directory
cd landing-page

# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Start production server
npm start
```

### Environment Variables

Create a `.env.local` file in the landing-page directory:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_PLATFORM_URL=http://localhost:3000
```

## 📝 Configuration

### Constants

All site content is configured in `lib/constants.ts`:

- **NAV_LINKS**: Navigation menu items
- **FEATURES**: Feature cards with icons and descriptions
- **PRICING_PLANS**: Pricing tiers and features
- **STATS**: Statistics to display
- **TESTIMONIALS**: Customer testimonials
- **FOOTER_LINKS**: Footer link sections
- **SOCIAL_LINKS**: Social media links

### Colors

The color palette is defined in `tailwind.config.ts`:

- Background: `#020617` (Deep Slate)
- Surface: `#0f172a` (Slate 900)
- Primary: `#3b82f6` (Royal Blue)
- Accent: `#8b5cf6` (Violet)

## 🎨 Customization

### Theming

Modify `tailwind.config.ts` to customize colors, fonts, and animations.

### Content

Update `lib/constants.ts` to modify:
- Feature descriptions
- Pricing plans
- Testimonials
- Navigation links
- Footer content

## 📦 Deployment

### Vercel (Recommended)

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel
```

### Docker

```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

## 🔗 Integration

The landing page integrates with the main KERNELIZE Platform:

- **Login**: Links to `http://localhost:3000/login`
- **Sign Up**: Links to `http://localhost:3000/signup`
- **API URL**: Configurable via environment variables

## 📄 License

Copyright (c) 2026 KERNELIZE Platform. All rights reserved.
