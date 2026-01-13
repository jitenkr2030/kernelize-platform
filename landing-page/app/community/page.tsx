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
import { Zap, MessageSquare, Users, Github, Twitter, Linkedin, Globe, ArrowRight, Heart } from 'lucide-react'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import Link from 'next/link'

const communityStats = [
  { value: '50K+', label: 'Community Members' },
  { value: '10K+', label: 'Discord Members' },
  { value: '5K+', label: 'GitHub Stars' },
  { value: '500+', label: 'Contributors' },
]

const discussionCategories = [
  { name: 'General Discussion', description: 'General topics and announcements', count: 1234 },
  { name: 'Help & Support', description: 'Get help from the community', count: 3456 },
  { name: 'Show & Tell', description: 'Share your projects and integrations', count: 789 },
  { name: 'Feature Requests', description: 'Suggest new features and improvements', count: 456 },
  { name: 'Off Topic', description: 'Non-work banter and fun stuff', count: 234 },
]

const upcomingEvents = [
  {
    title: 'Community Meetup - January',
    date: 'January 25, 2026',
    time: '6:00 PM PST',
    type: 'Virtual',
    description: 'Monthly community gathering with team updates and demos.',
  },
  {
    title: 'Hackathon 2026',
    date: 'February 15-17, 2026',
    time: 'All Weekend',
    type: 'Hybrid',
    description: 'Three-day virtual hackathon with $10K in prizes.',
  },
  {
    title: 'Office Hours',
    date: 'Every Thursday',
    time: '10:00 AM PST',
    type: 'Virtual',
    description: 'Drop-in session with KERNELIZE engineers.',
  },
]

export default function CommunityPage() {
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
              Join Our Community
            </h1>
            <p className="text-lg text-text-secondary max-w-2xl mx-auto mb-8">
              Connect with thousands of developers building amazing things with KERNELIZE. Share knowledge, get help, and shape the future of our platform.
            </p>
            <div className="flex flex-wrap items-center justify-center gap-4">
              <a
                href="https://discord.gg/kernelize"
                target="_blank"
                rel="noopener noreferrer"
              >
                <Button size="lg">
                  <MessageSquare className="w-5 h-5 mr-2" />
                  Join Discord
                </Button>
              </a>
              <a
                href="https://github.com/kernelize"
                target="_blank"
                rel="noopener noreferrer"
              >
                <Button variant="secondary" size="lg">
                  <Github className="w-5 h-5 mr-2" />
                  GitHub
                </Button>
              </a>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Stats */}
      <section className="py-12 border-y border-slate-800">
        <div className="container-custom">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {communityStats.map((stat, index) => (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                className="text-center"
              >
                <div className="text-3xl md:text-4xl font-bold gradient-text mb-2">
                  {stat.value}
                </div>
                <div className="text-text-secondary">
                  {stat.label}
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Discussion Forums */}
      <section className="py-16">
        <div className="container-custom">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
            className="text-center mb-12"
          >
            <h2 className="text-3xl md:text-4xl font-bold text-text-primary mb-4">
              Discussion Forums
            </h2>
            <p className="text-text-secondary max-w-2xl mx-auto">
              Join the conversation in our community forums. Ask questions, share ideas, and connect with other developers.
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-5xl mx-auto">
            {discussionCategories.map((category, index) => (
              <motion.div
                key={category.name}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
              >
                <Link href={`/community/${category.name.toLowerCase().replace(/\s+/g, '-')}`}>
                  <Card variant="bordered" hover className="p-6 h-full group">
                    <h3 className="text-lg font-semibold text-text-primary mb-2 group-hover:text-primary transition-colors">
                      {category.name}
                    </h3>
                    <p className="text-text-secondary text-sm mb-4">
                      {category.description}
                    </p>
                    <div className="flex items-center justify-between">
                      <span className="text-text-secondary text-sm">
                        {category.count.toLocaleString()} topics
                      </span>
                      <ArrowRight className="w-5 h-5 text-text-secondary group-hover:text-primary group-hover:translate-x-1 transition-all" />
                    </div>
                  </Card>
                </Link>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Upcoming Events */}
      <section className="py-16">
        <div className="container-custom">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
            className="text-center mb-12"
          >
            <h2 className="text-3xl md:text-4xl font-bold text-text-primary mb-4">
              Upcoming Events
            </h2>
            <p className="text-text-secondary max-w-2xl mx-auto">
              Connect with the community at our upcoming events and meetups.
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-5xl mx-auto">
            {upcomingEvents.map((event, index) => (
              <motion.div
                key={event.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
              >
                <Card variant="bordered" className="p-6 h-full">
                  <span className="inline-block px-3 py-1 rounded-full text-xs font-medium bg-primary/10 text-primary mb-4">
                    {event.type}
                  </span>
                  <h3 className="text-lg font-semibold text-text-primary mb-2">
                    {event.title}
                  </h3>
                  <div className="flex items-center gap-2 text-sm text-text-secondary mb-3">
                    <Globe className="w-4 h-4" />
                    {event.date}
                  </div>
                  <div className="flex items-center gap-2 text-sm text-text-secondary mb-4">
                    <Zap className="w-4 h-4" />
                    {event.time}
                  </div>
                  <p className="text-text-secondary text-sm mb-4">
                    {event.description}
                  </p>
                  <Button variant="secondary" size="sm" className="w-full">
                    Register
                  </Button>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Community Showcase */}
      <section className="py-16">
        <div className="container-custom">
          <Card variant="default" className="p-8 md:p-12 text-center relative overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-br from-primary/10 via-accent/5 to-transparent" />
            <div className="relative z-10">
              <Users className="w-16 h-16 text-primary mx-auto mb-6" />
              <h2 className="text-3xl font-bold text-text-primary mb-4">
                Community Showcase
              </h2>
              <p className="text-text-secondary max-w-2xl mx-auto mb-8">
                See what our community is building with KERNELIZE. From startups to enterprise applications, discover inspiring projects from fellow developers.
              </p>
              <Button size="lg">
                View Projects
                <ArrowRight className="w-5 h-5 ml-2" />
              </Button>
            </div>
          </Card>
        </div>
      </section>

      {/* Connect With Us */}
      <section className="py-16">
        <div className="container-custom">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
            className="text-center mb-12"
          >
            <h2 className="text-3xl md:text-4xl font-bold text-text-primary mb-4">
              Connect With Us
            </h2>
            <p className="text-text-secondary max-w-2xl mx-auto">
              Follow us on social media for updates, tips, and community highlights.
            </p>
          </motion.div>

          <div className="flex flex-wrap items-center justify-center gap-6">
            {[
              { icon: Twitter, name: 'Twitter', href: 'https://twitter.com/kernelize' },
              { icon: Github, name: 'GitHub', href: 'https://github.com/kernelize' },
              { icon: Linkedin, name: 'LinkedIn', href: 'https://linkedin.com/company/kernelize' },
              { icon: MessageSquare, name: 'Discord', href: 'https://discord.gg/kernelize' },
            ].map((social, index) => (
              <motion.a
                key={social.name}
                href={social.href}
                target="_blank"
                rel="noopener noreferrer"
                initial={{ opacity: 0, scale: 0.9 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={{ duration: 0.3, delay: index * 0.1 }}
                className="flex items-center gap-3 px-6 py-4 bg-surface border border-slate-700 rounded-xl hover:border-primary transition-colors group"
              >
                <social.icon className="w-6 h-6 text-text-secondary group-hover:text-primary transition-colors" />
                <span className="font-medium text-text-primary">{social.name}</span>
              </motion.a>
            ))}
          </div>
        </div>
      </section>

      {/* Contribution */}
      <section className="py-16 pb-24">
        <div className="container-custom">
          <div className="max-w-3xl mx-auto text-center">
            <Heart className="w-12 h-12 text-red-500 mx-auto mb-6" />
            <h2 className="text-3xl font-bold text-text-primary mb-4">
              Open Source Contributions
            </h2>
            <p className="text-text-secondary mb-8">
              KERNELIZE is built in the open. Our SDKs, libraries, and tools are all open source. We welcome contributions from the community!
            </p>
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <a
                href="https://github.com/kernelize"
                target="_blank"
                rel="noopener noreferrer"
              >
                <Button size="lg">
                  Contribute on GitHub
                </Button>
              </a>
              <Link href="/docs/contributing">
                <Button variant="secondary" size="lg">
                  Contribution Guide
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </section>
    </main>
  )
}
