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
import { Zap, ArrowUpRight, Shield, Globe, Database, Cloud, Server, Users, Cpu, Lock, Scale } from 'lucide-react'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import Link from 'next/link'

const features = [
  {
    title: 'High Availability',
    description: '99.99% uptime guarantee with redundant zones. Our distributed architecture ensures your applications infrastructure across multiple availability stay online even during hardware failures or zone outages.',
    icon: Server,
    gradient: 'from-green-500 to-emerald-600',
    benefits: ['Multi-AZ deployment', 'Automatic failover', 'Health monitoring', 'SLA guarantee'],
  },
  {
    title: 'Auto Scaling',
    description: 'Automatically scale your resources based on demand with intelligent load balancing. Handle traffic spikes effortlessly without overprovisioning or manual intervention.',
    icon: Scale,
    gradient: 'from-blue-500 to-cyan-600',
    benefits: ['Horizontal scaling', 'Custom policies', 'Predictive scaling', 'Cost optimization'],
  },
  {
    title: 'Enterprise Security',
    description: 'Bank-grade encryption, SOC 2 compliance, and advanced threat detection built-in. Protect your data with industry-leading security measures.',
    icon: Lock,
    gradient: 'from-purple-500 to-violet-600',
    benefits: ['End-to-end encryption', 'SOC 2 Type II certified', 'WAF & DDoS protection', 'Audit logs'],
  },
  {
    title: 'Global CDN',
    description: 'Deliver content from edge locations worldwide with sub-100ms latency. Accelerate your applications and reduce load times for users globally.',
    icon: Globe,
    gradient: 'from-orange-500 to-amber-600',
    benefits: ['200+ edge locations', 'Automatic optimization', 'Image compression', 'Real-time caching'],
  },
  {
    title: 'Data Pipeline',
    description: 'Streamline your data workflows with our powerful ETL engine and real-time processing. Transform, clean, and analyze data at scale.',
    icon: Database,
    gradient: 'from-pink-500 to-rose-600',
    benefits: ['Real-time streaming', 'Batch processing', 'Data validation', 'Schema management'],
  },
  {
    title: 'Cloud Integration',
    description: 'Seamlessly integrate with AWS, Azure, Google Cloud, and other providers. Build hybrid and multi-cloud architectures with ease.',
    icon: Cloud,
    gradient: 'from-indigo-500 to-blue-600',
    benefits: ['Multi-cloud support', 'Native APIs', 'Pre-built connectors', 'Unified management'],
  },
]

const advancedFeatures = [
  { icon: Zap, title: 'Lightning Fast', desc: 'Sub-millisecond response times' },
  { icon: Cpu, title: 'Edge Computing', desc: 'Process data at the edge' },
  { icon: Users, title: 'Team Collaboration', desc: 'Built-in workflows and RBAC' },
  { icon: Shield, title: 'Compliance Ready', desc: 'GDPR, HIPAA, and more' },
]

export default function FeaturesPage() {
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
              Powerful Features for
              <br />
              <span className="gradient-text">Modern Applications</span>
            </h1>
            <p className="text-lg text-text-secondary max-w-2xl mx-auto mb-8">
              Everything you need to build, deploy, and scale your applications with confidence.
              Enterprise-grade infrastructure meets developer-friendly tools.
            </p>
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <Link href="/signup">
                <Button size="lg">
                  Start Free Trial
                </Button>
              </Link>
              <Link href="/contact">
                <Button variant="secondary" size="lg">
                  Contact Sales
                </Button>
              </Link>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Main Features */}
      <section className="py-16">
        <div className="container-custom">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
              >
                <Card variant="bordered" hover className="h-full p-6 group">
                  <div className={`w-14 h-14 rounded-xl bg-gradient-to-br ${feature.gradient} flex items-center justify-center mb-4 group-hover:scale-110 transition-transform duration-300`}>
                    <feature.icon className="w-7 h-7 text-white" />
                  </div>
                  <h3 className="text-xl font-semibold text-text-primary mb-3">
                    {feature.title}
                  </h3>
                  <p className="text-text-secondary mb-4">
                    {feature.description}
                  </p>
                  <ul className="space-y-2">
                    {feature.benefits.map((benefit) => (
                      <li key={benefit} className="flex items-center gap-2 text-sm text-text-secondary">
                        <Zap className="w-4 h-4 text-primary flex-shrink-0" />
                        {benefit}
                      </li>
                    ))}
                  </ul>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Advanced Features */}
      <section className="py-16 relative">
        <div className="absolute inset-0 bg-surface/50" />
        <div className="container-custom relative z-10">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
            className="text-center mb-12"
          >
            <h2 className="text-3xl md:text-4xl font-bold text-text-primary mb-4">
              And Much More
            </h2>
            <p className="text-text-secondary max-w-2xl mx-auto">
              Additional capabilities that make KERNELIZE the complete platform for your needs.
            </p>
          </motion.div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            {advancedFeatures.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
              >
                <Card variant="default" className="p-6 text-center h-full">
                  <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center mx-auto mb-4">
                    <feature.icon className="w-6 h-6 text-primary" />
                  </div>
                  <h3 className="font-semibold text-text-primary mb-2">
                    {feature.title}
                  </h3>
                  <p className="text-sm text-text-secondary">
                    {feature.desc}
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
          <Card variant="bordered" className="p-8 md:p-12 text-center relative overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-br from-primary/10 via-accent/5 to-transparent" />
            <div className="relative z-10">
              <h2 className="text-3xl md:text-4xl font-bold text-text-primary mb-4">
                Ready to Explore All Features?
              </h2>
              <p className="text-text-secondary max-w-2xl mx-auto mb-8">
                Start your free trial today and experience the full power of KERNELIZE Platform.
              </p>
              <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
                <Link href="/signup">
                  <Button size="lg">
                    Start Free Trial
                    <ArrowUpRight className="w-5 h-5 ml-2" />
                  </Button>
                </Link>
                <Link href="/docs">
                  <Button variant="secondary" size="lg">
                    View Documentation
                  </Button>
                </Link>
              </div>
            </div>
          </Card>
        </div>
      </section>
    </main>
  )
}
