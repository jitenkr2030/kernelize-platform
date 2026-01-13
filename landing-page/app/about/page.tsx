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
import { Zap, Shield, Globe, Cpu, Users, Award, Target, Heart } from 'lucide-react'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import Link from 'next/link'

const team = [
  {
    name: 'Sarah Chen',
    role: 'CEO & Co-founder',
    bio: 'Former VP of Engineering at CloudScale. 15+ years in distributed systems.',
    image: '/team/sarah.jpg',
  },
  {
    name: 'Michael Rodriguez',
    role: 'CTO & Co-founder',
    bio: 'Infrastructure architect who previously led teams at Google and AWS.',
    image: '/team/michael.jpg',
  },
  {
    name: 'Emily Watson',
    role: 'VP of Product',
    bio: 'Product leader with experience at Stripe and Shopify. Passionate about developer experience.',
    image: '/team/emily.jpg',
  },
  {
    name: 'David Kim',
    role: 'VP of Engineering',
    bio: 'Open source contributor and former maintainer of several CNCF projects.',
    image: '/team/david.jpg',
  },
]

const values = [
  {
    icon: Target,
    title: 'Developer First',
    description: 'We build tools that developers love. Every decision starts with the question: Will this make developers more productive?',
  },
  {
    icon: Shield,
    title: 'Security by Design',
    description: 'Security isn&apos;t an afterthought. It&apos;s built into every layer of our infrastructure from the ground up.',
  },
  {
    icon: Globe,
    title: 'Global Scale',
    description: 'We believe world-class infrastructure should be accessible to everyone, everywhere, regardless of company size.',
  },
  {
    icon: Heart,
    title: 'Open Source',
    description: 'We&apos;re committed to the open source community and contribute back to the projects we use and love.',
  },
]

const stats = [
  { value: '500K+', label: 'Active Developers' },
  { value: '50M+', label: 'API Requests/Day' },
  { value: '99.99%', label: 'Uptime SLA' },
  { value: '150+', label: 'Countries' },
]

export default function AboutPage() {
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
              Building the Future of
              <br />
              <span className="gradient-text">Cloud Infrastructure</span>
            </h1>
            <p className="text-lg text-text-secondary max-w-2xl mx-auto mb-8">
              We&apos;re on a mission to make enterprise-grade infrastructure accessible to every developer and organization.
            </p>
          </motion.div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-12 border-y border-slate-800">
        <div className="container-custom">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {stats.map((stat, index) => (
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

      {/* Mission Section */}
      <section className="py-16">
        <div className="container-custom">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5 }}
            >
              <h2 className="text-3xl md:text-4xl font-bold text-text-primary mb-6">
                Our Mission
              </h2>
              <p className="text-text-secondary mb-4">
                Founded in 2023, KERNELIZE emerged from a simple observation: building and scaling modern applications was still too hard. Too many companies were spending millions on infrastructure teams instead of building their core products.
              </p>
              <p className="text-text-secondary mb-4">
                We believed there had to be a better way. A platform that combines the best of cloud-native technologies with developer-friendly tools, making it possible for teams of any size to deploy production-ready infrastructure in minutes, not months.
              </p>
              <p className="text-text-secondary mb-6">
                Today, KERNELIZE powers applications for thousands of companies worldwide, from startups to Fortune 500 enterprises. And we&apos;re just getting started.
              </p>
              <Link href="/contact">
                <Button size="lg">
                  Get in Touch
                </Button>
              </Link>
            </motion.div>
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
            >
              <div className="relative">
                <div className="absolute inset-0 bg-gradient-to-br from-primary/20 to-accent/20 rounded-3xl blur-3xl" />
                <div className="relative bg-surface/50 rounded-3xl p-8 border border-slate-700">
                  <Zap className="w-16 h-16 text-primary mb-6" />
                  <h3 className="text-2xl font-bold text-text-primary mb-4">
                    From Idea to Production in Minutes
                  </h3>
                  <p className="text-text-secondary">
                    Our platform abstracts away the complexity of distributed systems, giving developers the power to focus on what matters most: building great products.
                  </p>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Values Section */}
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
              Our Values
            </h2>
            <p className="text-text-secondary max-w-2xl mx-auto">
              The principles that guide everything we do at KERNELIZE.
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {values.map((value, index) => (
              <motion.div
                key={value.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
              >
                <Card variant="bordered" hover className="p-6 h-full text-center">
                  <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center mx-auto mb-4">
                    <value.icon className="w-6 h-6 text-primary" />
                  </div>
                  <h3 className="text-lg font-semibold text-text-primary mb-3">
                    {value.title}
                  </h3>
                  <p className="text-text-secondary text-sm">
                    {value.description}
                  </p>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Team Section */}
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
              Meet Our Team
            </h2>
            <p className="text-text-secondary max-w-2xl mx-auto">
              The people behind KERNELIZE, united by a shared passion for building great tools.
            </p>
          </motion.div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 max-w-5xl mx-auto">
            {team.map((member, index) => (
              <motion.div
                key={member.name}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
              >
                <Card variant="bordered" hover className="p-6 h-full text-center">
                  <div className="w-20 h-20 rounded-full bg-gradient-to-br from-primary to-accent flex items-center justify-center mx-auto mb-4 text-white text-2xl font-bold">
                    {member.name.split(' ').map(n => n[0]).join('')}
                  </div>
                  <h3 className="text-lg font-semibold text-text-primary mb-1">
                    {member.name}
                  </h3>
                  <p className="text-primary text-sm mb-3">
                    {member.role}
                  </p>
                  <p className="text-text-secondary text-sm">
                    {member.bio}
                  </p>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-16">
        <div className="container-custom">
          <Card variant="default" className="p-8 md:p-12 text-center relative overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-br from-primary/10 via-accent/5 to-transparent" />
            <div className="relative z-10">
              <h2 className="text-3xl font-bold text-text-primary mb-4">
                Join Us on Our Mission
              </h2>
              <p className="text-text-secondary max-w-2xl mx-auto mb-8">
                We&apos;re always looking for talented people who share our passion for building great tools.
              </p>
              <Link href="/careers">
                <Button size="lg">
                  View Open Positions
                </Button>
              </Link>
            </div>
          </Card>
        </div>
      </section>
    </main>
  )
}
