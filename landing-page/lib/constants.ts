/**
 * KERNELIZE Platform - Landing Page
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { 
  Zap, 
  ArrowUpRight, 
  Shield, 
  Globe, 
  Database, 
  Cloud,
  Server,
  Users,
  Globe2
} from 'lucide-react'

// Platform name and branding
export const PLATFORM_NAME = 'KERNELIZE'
export const PLATFORM_TAGLINE = 'Enterprise-Grade Infrastructure Platform'
export const PLATFORM_DESCRIPTION = 'Build, deploy, and scale your applications with our comprehensive cloud infrastructure solution.'

// Navigation links
export const NAV_LINKS = [
  { label: 'Features', href: '/features' },
  { label: 'Pricing', href: '/pricing' },
  { label: 'Documentation', href: '/docs' },
  { label: 'Resources', href: '/resources/api' },
]

// Social media links
export const SOCIAL_LINKS = [
  { name: 'Twitter', href: 'https://twitter.com/kernelize', icon: 'Twitter' },
  { name: 'GitHub', href: 'https://github.com/kernelize', icon: 'GitHub' },
  { name: 'LinkedIn', href: 'https://linkedin.com/company/kernelize', icon: 'LinkedIn' },
  { name: 'Discord', href: 'https://discord.gg/kernelize', icon: 'Discord' },
]

// Footer links organized by category
export const FOOTER_LINKS = {
  Product: [
    { label: 'Features', href: '/features' },
    { label: 'Pricing', href: '/pricing' },
    { label: 'Documentation', href: '/docs' },
    { label: 'Changelog', href: '/changelog' },
  ],
  Company: [
    { label: 'About Us', href: '/about' },
    { label: 'Blog', href: '/blog' },
    { label: 'Careers', href: '/careers' },
    { label: 'Contact', href: '/contact' },
  ],
  Legal: [
    { label: 'Privacy Policy', href: '/privacy' },
    { label: 'Terms of Service', href: '/terms' },
    { label: 'Cookie Policy', href: '/cookie-policy' },
  ],
  Resources: [
    { label: 'API Reference', href: '/resources/api' },
    { label: 'SDK Downloads', href: '/resources/sdk' },
    { label: 'Community', href: '/community' },
    { label: 'Support', href: '/support' },
  ],
}

// Platform features
export const FEATURES = [
  {
    title: 'High Availability',
    description: '99.99% uptime guarantee with redundant infrastructure across multiple availability zones.',
    icon: Zap,
    gradient: 'from-green-500 to-emerald-600',
  },
  {
    title: 'Auto Scaling',
    description: 'Automatically scale your resources based on demand with intelligent load balancing.',
    icon: ArrowUpRight,
    gradient: 'from-blue-500 to-cyan-600',
  },
  {
    title: 'Enterprise Security',
    description: 'Bank-grade encryption, SOC 2 compliance, and advanced threat detection built-in.',
    icon: Shield,
    gradient: 'from-purple-500 to-violet-600',
  },
  {
    title: 'Global CDN',
    description: 'Deliver content from edge locations worldwide with sub-100ms latency.',
    icon: Globe,
    gradient: 'from-orange-500 to-amber-600',
  },
  {
    title: 'Data Pipeline',
    description: 'Streamline your data workflows with our powerful ETL engine and real-time processing.',
    icon: Database,
    gradient: 'from-pink-500 to-rose-600',
  },
  {
    title: 'Cloud Integration',
    description: 'Seamlessly integrate with AWS, Azure, Google Cloud, and other providers.',
    icon: Cloud,
    gradient: 'from-indigo-500 to-blue-600',
  },
]

// Statistics to display
export const STATS = [
  { value: '99.99', label: 'Uptime SLA', icon: Server, suffix: '%' },
  { value: '50M+', label: 'API Requests/Day', icon: Globe2, suffix: '' },
  { value: '500K+', label: 'Active Developers', icon: Users, suffix: '' },
  { value: '150+', label: 'Countries Served', icon: Globe, suffix: '' },
]

// Pricing plans
export const PRICING_PLANS = [
  {
    name: 'Starter',
    price: 299,
    description: 'Perfect for small projects and individual developers',
    features: [
      'Up to 100GB bandwidth',
      '10 custom domains',
      'Basic analytics',
      'Email support',
      'Community access',
    ],
    cta: 'Start Free Trial',
    popular: false,
    gradient: 'from-slate-500 to-slate-600',
  },
  {
    name: 'Professional',
    price: 499,
    description: 'Ideal for growing teams and production applications',
    features: [
      'Up to 1TB bandwidth',
      'Unlimited custom domains',
      'Advanced analytics & reports',
      'Priority email & chat support',
      'Team collaboration tools',
      'Custom integrations',
    ],
    cta: 'Get Started',
    popular: true,
    gradient: 'from-blue-500 to-indigo-600',
  },
  {
    name: 'Enterprise',
    price: 999999,
    description: 'For large organizations with advanced requirements',
    features: [
      'Unlimited bandwidth',
      'Unlimited custom domains',
      'Custom analytics & dashboards',
      '24/7 phone & dedicated support',
      'SLA guarantee',
      'Custom contracts',
      'On-premise deployment option',
      'Training & onboarding',
    ],
    cta: 'Contact Sales',
    popular: false,
    gradient: 'from-purple-500 to-violet-600',
  },
]

// Hero section content
export const HERO_CONTENT = {
  headline: 'Build Faster, Scale Better',
  subheadline: 'The complete infrastructure platform for modern applications. Deploy in seconds, scale to millions.',
  ctaPrimary: 'Start Building Free',
  ctaSecondary: 'Watch Demo',
}

// Company contact information
export const CONTACT_INFO = {
  email: 'contact@kernelize.com',
  phone: '+1 (555) 123-4567',
  address: '100 Technology Drive, San Francisco, CA 94105',
}

// Customer testimonials
export const TESTIMONIALS = [
  {
    content: "KERNELIZE has transformed how we deploy and scale our applications. We've reduced our deployment time by 80%.",
    author: "Sarah Chen",
    role: "CTO",
    company: "TechFlow Inc.",
    avatar: "/testimonials/sarah.jpg",
    rating: 5,
  },
  {
    content: "The enterprise security features gave us the confidence to migrate our entire infrastructure to the cloud.",
    author: "Michael Rodriguez",
    role: "VP of Engineering",
    company: "SecureNet Solutions",
    avatar: "/testimonials/michael.jpg",
    rating: 5,
  },
  {
    content: "Outstanding support team and seamless scaling capabilities. KERNELIZE is now essential to our business.",
    author: "Emily Watson",
    role: "Lead Developer",
    company: "CloudFirst Labs",
    avatar: "/testimonials/emily.jpg",
    rating: 5,
  },
]
