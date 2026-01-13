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
import { ArrowRight, Zap, Shield, Cpu, Database, Cloud } from 'lucide-react'
import { Button } from '@/components/ui/Button'
import Link from 'next/link'

export default function Hero() {
  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden pt-20">
      {/* Background Effects */}
      <div className="absolute inset-0 bg-gradient-to-br from-primary/5 via-transparent to-accent/5" />
      <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-primary/20 rounded-full blur-3xl animate-pulse" />
      <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-accent/20 rounded-full blur-3xl animate-pulse delay-1000" />

      <div className="container-custom relative z-10">
        <div className="text-center max-w-5xl mx-auto">
          {/* Badge */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass mb-8"
          >
            <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
            <span className="text-sm text-text-secondary">Now with AI-Powered Compression 2.0</span>
          </motion.div>

          {/* Main Heading */}
          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="text-4xl sm:text-5xl lg:text-7xl font-bold mb-6"
          >
            <span className="text-text-primary">Enterprise AI & Data</span>
            <br />
            <span className="gradient-text">Management Platform</span>
          </motion.h1>

          {/* Subtitle */}
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="text-lg sm:text-xl text-text-secondary max-w-3xl mx-auto mb-10"
          >
            Comprehensive enterprise-grade platform implementing advanced AI compression, 
            analytics, data management, and DevOps operations. Deploy production-ready in minutes.
          </motion.p>

          {/* CTA Buttons */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
            className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-16"
          >
            <Link href="/signup">
              <Button size="lg" className="group w-full sm:w-auto">
                Start Free Trial
                <ArrowRight className="w-5 h-5 ml-2 group-hover:translate-x-1 transition-transform" />
              </Button>
            </Link>
            <Link href="#features">
              <Button variant="secondary" size="lg" className="w-full sm:w-auto">
                Explore Features
              </Button>
            </Link>
          </motion.div>

          {/* Stats */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.4 }}
            className="grid grid-cols-2 md:grid-cols-4 gap-4 max-w-4xl mx-auto"
          >
            {[
              { icon: Database, value: '50PB+', label: 'Data Processed' },
              { icon: Cloud, value: '99.9%', label: 'Uptime SLA' },
              { icon: Cpu, value: '10K+', label: 'API Calls/Sec' },
              { icon: Shield, value: '500+', label: 'Enterprise Users' },
            ].map((stat, index) => (
              <div
                key={stat.label}
                className="glass rounded-xl p-4 card-hover"
              >
                <stat.icon className="w-8 h-8 text-primary mx-auto mb-2" />
                <div className="text-2xl font-bold text-text-primary">{stat.value}</div>
                <div className="text-sm text-text-secondary">{stat.label}</div>
              </div>
            ))}
          </motion.div>

          {/* Visual Element */}
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.5 }}
            className="mt-16 relative"
          >
            <div className="absolute inset-0 bg-gradient-to-t from-background via-transparent to-transparent z-10" />
            <div className="glass rounded-2xl border border-slate-700 overflow-hidden">
              <div className="bg-surface/50 px-4 py-3 flex items-center gap-2 border-b border-slate-700">
                <div className="w-3 h-3 rounded-full bg-red-500" />
                <div className="w-3 h-3 rounded-full bg-yellow-500" />
                <div className="w-3 h-3 rounded-full bg-green-500" />
                <span className="text-text-secondary text-sm ml-2">kernelize-platform</span>
              </div>
              <div className="p-6 text-left font-mono text-sm overflow-x-auto">
                <div className="text-text-secondary">
                  <span className="text-primary">$</span> kernelize init my-project
                </div>
                <div className="text-green-400 mt-2">
                  ✓ Project initialized successfully
                </div>
                <div className="text-text-secondary mt-2">
                  <span className="text-primary">$</span> kernelize deploy --production
                </div>
                <div className="text-green-400 mt-2">
                  ✓ Infrastructure deployed (Terraform)<br />
                  ✓ Kubernetes cluster configured<br />
                  ✓ CI/CD pipeline activated<br />
                  ✓ Monitoring stack deployed<br />
                  ✓ Production URL: https://app.kernelize.platform
                </div>
                <div className="text-text-secondary mt-2">
                  <span className="text-primary">$</span> kernelize status
                </div>
                <div className="text-green-400 mt-2">
                  ✓ All services healthy<br />
                  ✓ 99.99% uptime achieved<br />
                  ✓ Data processed: 1.2TB today
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </div>

      {/* Decorative Elements */}
      <div className="absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-t from-background to-transparent" />
    </section>
  )
}
