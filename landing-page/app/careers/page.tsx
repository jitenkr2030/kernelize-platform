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
import { Zap, ArrowRight, MapPin, Users, Clock, DollarSign, Heart, Brain, Shield, Globe } from 'lucide-react'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import Link from 'next/link'
import { useState } from 'react'

const benefits = [
  {
    icon: Heart,
    title: 'Health & Wellness',
    description: 'Comprehensive medical, dental, and vision coverage for you and your family.',
  },
  {
    icon: Brain,
    title: 'Learning & Development',
    description: '$2,500 annual learning budget for courses, conferences, and books.',
  },
  {
    icon: Globe,
    title: 'Flexible Time Off',
    description: 'Unlimited PTO with encouraged minimum of 4 weeks per year.',
  },
  {
    icon: Users,
    title: 'Remote-First',
    description: 'Work from anywhere. We&apos;re a distributed team with offices in SF and NYC.',
  },
  {
    icon: DollarSign,
    title: 'Competitive Compensation',
    description: 'Top-tier salaries with significant equity packages for all employees.',
  },
  {
    icon: Shield,
    title: 'Retirement',
    description: '401(k) with 4% company match, immediately vested.',
  },
]

const openPositions = [
  {
    title: 'Senior Software Engineer, Platform',
    department: 'Engineering',
    location: 'San Francisco, CA (Hybrid)',
    type: 'Full-time',
    salary: '$180K - $220K',
  },
  {
    title: 'Software Engineer, Developer Experience',
    department: 'Engineering',
    location: 'Remote (US/EU)',
    type: 'Full-time',
    salary: '$140K - $180K',
  },
  {
    title: 'Product Manager, Infrastructure',
    department: 'Product',
    location: 'New York, NY (Hybrid)',
    type: 'Full-time',
    salary: '$160K - $200K',
  },
  {
    title: 'Solutions Architect',
    department: 'Sales',
    location: 'Remote (US)',
    type: 'Full-time',
    salary: '$150K - $190K + OTE',
  },
  {
    title: 'Security Engineer',
    department: 'Engineering',
    location: 'San Francisco, CA (Hybrid)',
    type: 'Full-time',
    salary: '$170K - $210K',
  },
  {
    title: 'Technical Writer',
    department: 'Product',
    location: 'Remote (Worldwide)',
    type: 'Full-time',
    salary: '$100K - $140K',
  },
]

export default function CareersPage() {
  const [expandedDept, setExpandedDept] = useState<string | null>('Engineering')

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
              Join Our Team
            </h1>
            <p className="text-lg text-text-secondary max-w-2xl mx-auto mb-8">
              We&apos;re building the future of cloud infrastructure. Come help us make enterprise-grade tools accessible to everyone.
            </p>
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <Link href="#positions">
                <Button size="lg">
                  View Open Positions
                  <ArrowRight className="w-5 h-5 ml-2" />
                </Button>
              </Link>
              <Link href="/about">
                <Button variant="secondary" size="lg">
                  Learn About Us
                </Button>
              </Link>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Why Join Us */}
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
              Why Join KERNELIZE?
            </h2>
            <p className="text-text-secondary max-w-2xl mx-auto">
              We offer competitive benefits and a supportive environment for you to do your best work.
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {benefits.map((benefit, index) => (
              <motion.div
                key={benefit.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
              >
                <Card variant="bordered" hover className="p-6 h-full">
                  <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center mb-4">
                    <benefit.icon className="w-6 h-6 text-primary" />
                  </div>
                  <h3 className="text-lg font-semibold text-text-primary mb-3">
                    {benefit.title}
                  </h3>
                  <p className="text-text-secondary text-sm">
                    {benefit.description}
                  </p>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Open Positions */}
      <section id="positions" className="py-16">
        <div className="container-custom">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
            className="text-center mb-12"
          >
            <h2 className="text-3xl md:text-4xl font-bold text-text-primary mb-4">
              Open Positions
            </h2>
            <p className="text-text-secondary max-w-2xl mx-auto">
              Find the role that&apos;s right for you and help us build the future of cloud infrastructure.
            </p>
          </motion.div>

          <div className="max-w-4xl mx-auto space-y-4">
            {openPositions.map((position, index) => (
              <motion.div
                key={position.title}
                initial={{ opacity: 0, y: 10 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.3, delay: index * 0.05 }}
              >
                <Link href={`/careers/${position.title.toLowerCase().replace(/\s+/g, '-')}`}>
                  <Card variant="bordered" hover className="p-6">
                    <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                      <div className="flex-1">
                        <h3 className="text-lg font-semibold text-text-primary mb-2">
                          {position.title}
                        </h3>
                        <div className="flex flex-wrap items-center gap-4 text-sm text-text-secondary">
                          <span className="flex items-center gap-1">
                            <Users className="w-4 h-4" />
                            {position.department}
                          </span>
                          <span className="flex items-center gap-1">
                            <MapPin className="w-4 h-4" />
                            {position.location}
                          </span>
                          <span className="flex items-center gap-1">
                            <Clock className="w-4 h-4" />
                            {position.type}
                          </span>
                          <span className="flex items-center gap-1">
                            <DollarSign className="w-4 h-4" />
                            {position.salary}
                          </span>
                        </div>
                      </div>
                      <div className="flex-shrink-0">
                        <Button variant="secondary" size="sm">
                          Apply Now
                          <ArrowRight className="w-4 h-4 ml-2" />
                        </Button>
                      </div>
                    </div>
                  </Card>
                </Link>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Values Section */}
      <section className="py-16">
        <div className="container-custom">
          <Card variant="default" className="p-8 md:p-12 text-center relative overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-br from-primary/10 via-accent/5 to-transparent" />
            <div className="relative z-10">
              <h2 className="text-3xl font-bold text-text-primary mb-4">
                Don&apos;t See the Right Role?
              </h2>
              <p className="text-text-secondary max-w-2xl mx-auto mb-8">
                We&apos;re always looking for talented people. Send us your resume and we&apos;ll keep you in mind for future opportunities.
              </p>
              <Link href="/contact">
                <Button size="lg">
                  Get in Touch
                </Button>
              </Link>
            </div>
          </Card>
        </div>
      </section>

      {/* EEO Statement */}
      <section className="py-8">
        <div className="container-custom">
          <p className="text-text-secondary text-sm text-center max-w-3xl mx-auto">
            KERNELIZE is an equal opportunity employer. We celebrate diversity and are committed to creating an inclusive environment for all employees. All employment decisions are based on qualifications, merit, and business needs.
          </p>
        </div>
      </section>
    </main>
  )
}
