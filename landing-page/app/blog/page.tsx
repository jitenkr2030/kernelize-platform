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

'use client'

import { motion } from 'framer-motion'
import { Zap, ArrowRight, Calendar, Users, Tag, ChevronRight } from 'lucide-react'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import Link from 'next/link'

const blogPosts = [
  {
    slug: 'introducing-ai-compression-2',
    title: 'Introducing AI-Powered Compression 2.0',
    excerpt: 'We\'re excited to announce our latest compression technology that can reduce storage costs by up to 70% while maintaining perfect data integrity.',
    author: 'Sarah Chen',
    role: 'CEO',
    date: 'January 10, 2026',
    readTime: '5 min read',
    category: 'Product',
    tags: ['Compression', 'AI', 'Storage'],
    gradient: 'from-blue-500 to-cyan-600',
  },
  {
    slug: 'scaling-to-10m-users',
    title: 'How We Scaled to 10 Million Users in 6 Months',
    excerpt: 'A deep dive into the technical challenges we faced and how our architecture evolved to support massive scale while maintaining performance.',
    author: 'Michael Rodriguez',
    role: 'CTO',
    date: 'December 28, 2025',
    readTime: '12 min read',
    category: 'Engineering',
    tags: ['Scaling', 'Architecture', 'Performance'],
    gradient: 'from-purple-500 to-violet-600',
  },
  {
    slug: 'security-best-practices-2026',
    title: 'Security Best Practices for 2026',
    excerpt: 'The threat landscape is constantly evolving. Here are our top recommendations for keeping your applications and data secure in the coming year.',
    author: 'David Kim',
    role: 'VP of Engineering',
    date: 'December 15, 2025',
    readTime: '8 min read',
    category: 'Security',
    tags: ['Security', 'Best Practices', 'Compliance'],
    gradient: 'from-green-500 to-emerald-600',
  },
  {
    slug: 'introducing-edge-computing',
    title: 'Introducing Edge Computing Capabilities',
    excerpt: 'Process data closer to your users with our new edge computing features. Reduce latency and improve performance for globally distributed applications.',
    author: 'Emily Watson',
    role: 'VP of Product',
    date: 'November 30, 2025',
    readTime: '6 min read',
    category: 'Product',
    tags: ['Edge', 'Performance', 'Global'],
    gradient: 'from-orange-500 to-amber-600',
  },
  {
    slug: 'kubernetes-at-scale',
    title: 'Running Kubernetes at Scale: Lessons Learned',
    excerpt: 'After managing thousands of clusters, we\'ve learned a thing or two. Here are our top insights for running Kubernetes in production.',
    author: 'Michael Rodriguez',
    role: 'CTO',
    date: 'November 15, 2025',
    readTime: '10 min read',
    category: 'Engineering',
    tags: ['Kubernetes', 'DevOps', 'Scale'],
    gradient: 'from-pink-500 to-rose-600',
  },
  {
    slug: 'building-developer-experience',
    title: 'The Art of Building Great Developer Experience',
    excerpt: 'Developer experience isn\'t just about documentation. It\'s about understanding the mental model of your users and designing for their success.',
    author: 'Emily Watson',
    role: 'VP of Product',
    date: 'November 1, 2025',
    readTime: '7 min read',
    category: 'Product',
    tags: ['DX', 'Design', 'Developer'],
    gradient: 'from-indigo-500 to-blue-600',
  },
]

const categories = ['All', 'Product', 'Engineering', 'Security', 'Company']

export default function BlogPage() {
  return (
    <main className="min-h-screen">
      {/* Hero Section */}
      <section className="pt-32 pb-16 relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-b from-primary/5 via-transparent to-transparent" />
        <div className="container-custom relative z-10">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="text-center max-w-4xl mx-auto"
          >
            <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold text-text-primary mb-6">
              Blog
            </h1>
            <p className="text-lg text-text-secondary max-w-2xl mx-auto mb-8">
              Insights, updates, and stories from the KERNELIZE team.
            </p>
          </motion.div>
        </div>
      </section>

      {/* Categories */}
      <section className="py-8">
        <div className="container-custom">
          <div className="flex flex-wrap items-center justify-center gap-4">
            {categories.map((category, index) => (
              <motion.button
                key={category}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3, delay: index * 0.05 }}
                className={`px-4 py-2 rounded-full text-sm font-medium transition-colors ${
                  category === 'All'
                    ? 'bg-primary text-white'
                    : 'bg-surface text-text-secondary hover:text-text-primary hover:bg-slate-700'
                }`}
              >
                {category}
              </motion.button>
            ))}
          </div>
        </div>
      </section>

      {/* Featured Post */}
      <section className="py-8">
        <div className="container-custom">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <Link href={`/blog/${blogPosts[0].slug}`}>
              <Card variant="default" className="overflow-hidden group cursor-pointer">
                <div className="grid grid-cols-1 lg:grid-cols-2">
                  <div className={`h-64 lg:h-auto bg-gradient-to-br ${blogPosts[0].gradient} flex items-center justify-center`}>
                    <Zap className="w-20 h-20 text-white opacity-50" />
                  </div>
                  <div className="p-8 md:p-12 flex flex-col justify-center">
                    <div className="flex items-center gap-2 mb-4">
                      <span className="px-3 py-1 rounded-full text-xs font-medium bg-primary/10 text-primary">
                        {blogPosts[0].category}
                      </span>
                      <span className="text-text-secondary text-sm flex items-center gap-1">
                        <Calendar className="w-4 h-4" />
                        {blogPosts[0].date}
                      </span>
                    </div>
                    <h2 className="text-2xl md:text-3xl font-bold text-text-primary mb-4 group-hover:text-primary transition-colors">
                      {blogPosts[0].title}
                    </h2>
                    <p className="text-text-secondary mb-6">
                      {blogPosts[0].excerpt}
                    </p>
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-full bg-gradient-to-br from-primary to-accent flex items-center justify-center text-white font-medium">
                          {blogPosts[0].author.split(' ').map(n => n[0]).join('')}
                        </div>
                        <div>
                          <div className="font-medium text-text-primary text-sm">
                            {blogPosts[0].author}
                          </div>
                          <div className="text-text-secondary text-xs">
                            {blogPosts[0].role}
                          </div>
                        </div>
                      </div>
                      <span className="text-text-secondary text-sm flex items-center gap-1">
                        {blogPosts[0].readTime}
                        <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                      </span>
                    </div>
                  </div>
                </div>
              </Card>
            </Link>
          </motion.div>
        </div>
      </section>

      {/* Blog Grid */}
      <section className="py-8 pb-16">
        <div className="container-custom">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {blogPosts.slice(1).map((post, index) => (
              <motion.div
                key={post.slug}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
              >
                <Link href={`/blog/${post.slug}`}>
                  <Card variant="bordered" hover className="h-full overflow-hidden group">
                    <div className={`h-48 bg-gradient-to-br ${post.gradient} flex items-center justify-center relative`}>
                      <Zap className="w-16 h-16 text-white opacity-30" />
                      <div className="absolute top-4 left-4">
                        <span className="px-3 py-1 rounded-full text-xs font-medium bg-white/20 text-white">
                          {post.category}
                        </span>
                      </div>
                    </div>
                    <div className="p-6">
                      <div className="flex items-center gap-2 mb-3">
                        <span className="text-text-secondary text-sm flex items-center gap-1">
                          <Calendar className="w-4 h-4" />
                          {post.date}
                        </span>
                        <span className="text-text-secondary text-sm">•</span>
                        <span className="text-text-secondary text-sm">
                          {post.readTime}
                        </span>
                      </div>
                      <h3 className="text-xl font-semibold text-text-primary mb-3 group-hover:text-primary transition-colors">
                        {post.title}
                      </h3>
                      <p className="text-text-secondary text-sm mb-4 line-clamp-2">
                        {post.excerpt}
                      </p>
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <div className="w-8 h-8 rounded-full bg-gradient-to-br from-primary to-accent flex items-center justify-center text-white text-xs font-medium">
                            {post.author.split(' ').map(n => n[0]).join('')}
                          </div>
                          <span className="text-text-secondary text-sm">
                            {post.author}
                          </span>
                        </div>
                        <ChevronRight className="w-5 h-5 text-text-secondary group-hover:text-primary group-hover:translate-x-1 transition-all" />
                      </div>
                    </div>
                  </Card>
                </Link>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Newsletter Section */}
      <section className="py-16">
        <div className="container-custom">
          <Card variant="default" className="p-8 md:p-12 text-center relative overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-br from-primary/10 via-accent/5 to-transparent" />
            <div className="relative z-10">
              <h2 className="text-3xl font-bold text-text-primary mb-4">
                Stay in the Loop
              </h2>
              <p className="text-text-secondary max-w-2xl mx-auto mb-8">
                Subscribe to our newsletter to get the latest posts delivered right to your inbox.
              </p>
              <div className="flex flex-col sm:flex-row items-center justify-center gap-4 max-w-md mx-auto">
                <input
                  type="email"
                  placeholder="Enter your email"
                  className="flex-1 px-4 py-3 bg-surface border border-slate-700 rounded-lg text-text-primary placeholder-text-secondary focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
                />
                <Button>
                  Subscribe
                </Button>
              </div>
            </div>
          </Card>
        </div>
      </section>
    </main>
  )
}
