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

import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { ArrowRight, Zap, Shield, Cpu, Database, Cloud, Terminal, Lock, Globe, Sparkles, Layers, Box, Cpu as CpuIcon } from 'lucide-react'
import { Button } from '@/components/ui/Button'
import Link from 'next/link'

// Floating particle component
function FloatingParticle({ delay, duration, size, startX, startY }: { delay: number; duration: number; size: number; startX: number; startY: number }) {
  return (
    <motion.div
      className="absolute rounded-full bg-primary/20"
      style={{
        width: size,
        height: size,
        left: `${startX}%`,
        top: `${startY}%`,
      }}
      animate={{
        y: [0, -100, 0],
        x: [0, 30, 0],
        opacity: [0, 0.6, 0],
      }}
      transition={{
        duration,
        delay,
        repeat: Infinity,
        ease: 'easeInOut',
      }}
    />
  )
}

export default function Hero() {
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 })

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      setMousePosition({
        x: (e.clientX / window.innerWidth) * 100,
        y: (e.clientY / window.innerHeight) * 100,
      })
    }
    window.addEventListener('mousemove', handleMouseMove)
    return () => window.removeEventListener('mousemove', handleMouseMove)
  }, [])

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
        delayChildren: 0.2,
      },
    },
  }

  const itemVariants = {
    hidden: { opacity: 0, y: 30 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        duration: 0.6,
        ease: 'easeOut',
      },
    },
  }

  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden pt-24">
      {/* Animated background orbs */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <motion.div
          className="orb orb-primary w-[600px] h-[600px] opacity-40"
          animate={{
            x: [0, 100, 0],
            y: [0, -50, 0],
          }}
          transition={{
            duration: 20,
            repeat: Infinity,
            ease: 'easeInOut',
          }}
          style={{
            left: '10%',
            top: '20%',
          }}
        />
        <motion.div
          className="orb orb-secondary w-[500px] h-[500px] opacity-30"
          animate={{
            x: [0, -80, 0],
            y: [0, 80, 0],
          }}
          transition={{
            duration: 25,
            repeat: Infinity,
            ease: 'easeInOut',
          }}
          style={{
            right: '10%',
            bottom: '20%',
          }}
        />
        <motion.div
          className="orb orb-accent w-[400px] h-[400px] opacity-25"
          animate={{
            x: [0, 50, 0],
            y: [0, -100, 0],
          }}
          transition={{
            duration: 18,
            repeat: Infinity,
            ease: 'easeInOut',
          }}
          style={{
            left: '50%',
            top: '40%',
          }}
        />
        
        {/* Grid pattern */}
        <div className="absolute inset-0 bg-grid opacity-20" />
        
        {/* Floating particles */}
        {[...Array(6)].map((_, i) => (
          <FloatingParticle
            key={i}
            delay={i * 0.5}
            duration={8 + i * 2}
            size={4 + Math.random() * 8}
            startX={20 + Math.random() * 60}
            startY={30 + Math.random() * 40}
          />
        ))}
      </div>

      <div className="container-custom relative z-10">
        <motion.div
          variants={containerVariants}
          initial="hidden"
          animate="visible"
          className="text-center max-w-5xl mx-auto"
        >
          {/* Badge */}
          <motion.div variants={itemVariants}>
            <Link href="/changelog" className="inline-flex items-center gap-3 px-6 py-3 glass rounded-full mb-8 hover:border-primary/30 transition-colors cursor-pointer group">
              <span className="relative flex h-3 w-3">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-accent opacity-75" />
                <span className="relative inline-flex rounded-full h-3 w-3 bg-accent" />
              </span>
              <span className="text-sm text-text-secondary group-hover:text-text-primary transition-colors">
                Introducing KERNELIZE 2.0 with AI-Powered Compression
              </span>
              <Sparkles className="w-4 h-4 text-accent" />
            </Link>
          </motion.div>

          {/* Main Heading */}
          <motion.h1
            variants={itemVariants}
            className="text-4xl sm:text-5xl lg:text-7xl font-bold mb-6 leading-tight"
          >
            <span className="text-text-primary">The Future of</span>
            <br />
            <span className="gradient-text">Enterprise AI Infrastructure</span>
          </motion.h1>

          {/* Subtitle */}
          <motion.p
            variants={itemVariants}
            className="text-lg sm:text-xl text-text-secondary max-w-3xl mx-auto mb-10 leading-relaxed"
          >
            Build, deploy, and scale next-generation AI applications with our comprehensive 
            platform featuring advanced compression, real-time analytics, and enterprise-grade security.
          </motion.p>

          {/* CTA Buttons */}
          <motion.div
            variants={itemVariants}
            className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-16"
          >
            <Link href="/signup">
              <Button size="lg" className="group w-full sm:w-auto text-lg px-8">
                <Zap className="w-5 h-5 mr-2" />
                Start Free Trial
                <ArrowRight className="w-5 h-5 ml-2 group-hover:translate-x-1 transition-transform" />
              </Button>
            </Link>
            <Link href="/docs">
              <Button variant="secondary" size="lg" className="w-full sm:w-auto text-lg px-8">
                <Terminal className="w-5 h-5 mr-2" />
                View Documentation
              </Button>
            </Link>
          </motion.div>

          {/* Bento Grid Stats */}
          <motion.div
            variants={itemVariants}
            className="grid grid-cols-2 lg:grid-cols-4 gap-4 max-w-4xl mx-auto mb-16"
          >
            {[
              { icon: Database, value: '50PB+', label: 'Data Processed', color: 'from-blue-500 to-cyan-500' },
              { icon: Cloud, value: '99.99%', label: 'Uptime SLA', color: 'from-green-500 to-emerald-500' },
              { icon: CpuIcon, value: '10K+', label: 'API Calls/Sec', color: 'from-purple-500 to-pink-500' },
              { icon: Globe, value: '150+', label: 'Countries', color: 'from-orange-500 to-amber-500' },
            ].map((stat, index) => (
              <motion.div
                key={stat.label}
                className="glass rounded-2xl p-6 text-center hover-border-gradient group cursor-default"
                whileHover={{ scale: 1.02 }}
                transition={{ duration: 0.3 }}
              >
                <div className={`w-12 h-12 mx-auto mb-4 rounded-xl bg-gradient-to-br ${stat.color} flex items-center justify-center opacity-80 group-hover:opacity-100 transition-opacity`}>
                  <stat.icon className="w-6 h-6 text-white" />
                </div>
                <div className="text-3xl font-bold gradient-text mb-1">{stat.value}</div>
                <div className="text-sm text-text-secondary">{stat.label}</div>
              </motion.div>
            ))}
          </motion.div>

          {/* Visual Element - Terminal */}
          <motion.div
            variants={itemVariants}
            className="relative max-w-5xl mx-auto"
          >
            {/* Terminal glow effect */}
            <div className="absolute inset-0 bg-gradient-to-t from-primary/20 via-transparent to-transparent rounded-2xl blur-2xl" />
            
            <div className="glass rounded-2xl border border-white/10 overflow-hidden relative">
              {/* Terminal header */}
              <div className="bg-surfaceLight/80 px-4 py-3 flex items-center gap-3 border-b border-white/5">
                <div className="flex gap-2">
                  <div className="w-3 h-3 rounded-full bg-red-500/80" />
                  <div className="w-3 h-3 rounded-full bg-yellow-500/80" />
                  <div className="w-3 h-3 rounded-full bg-green-500/80" />
                </div>
                <div className="flex-1 text-center">
                  <span className="text-text-muted text-sm font-mono">kernelize-terminal — v2.0</span>
                </div>
                <div className="flex items-center gap-2 text-text-muted text-xs">
                  <Lock className="w-3 h-3" />
                  <span>SSH Connected</span>
                </div>
              </div>
              
              {/* Terminal content */}
              <div className="p-6 text-left font-mono text-sm overflow-x-auto">
                <div className="text-text-secondary mb-4">
                  <span className="text-primary">$</span> kernelize init my-ai-project --template=enterprise
                </div>
                <div className="text-accent-light mt-2 mb-4">
                  <span className="text-green-400">✓</span> Project initialized successfully
                </div>
                <div className="text-text-secondary mb-2">
                  <span className="text-primary">$</span> kernelize deploy --production --scale=auto
                </div>
                <div className="text-green-400 mt-2 space-y-1">
                  <div>✓ Infrastructure provisioned (Terraform v1.6)</div>
                  <div>✓ Kubernetes cluster configured (3 nodes)</div>
                  <div>✓ AI model endpoints deployed (TensorFlow 2.14)</div>
                  <div>✓ CI/CD pipeline activated (GitHub Actions)</div>
                  <div>✓ Monitoring stack deployed (Prometheus + Grafana)</div>
                  <div>✓ CDN configured (50+ edge locations)</div>
                  <div className="mt-2 text-accent">● Production URL: https://app.kernelize.platform</div>
                </div>
                <div className="text-text-secondary mt-4">
                  <span className="text-primary">$</span> kernelize status --verbose
                </div>
                <div className="text-text-muted mt-2">
                  System Status: <span className="text-green-400">Healthy</span> | 
                  Latency: <span className="text-accent">23ms</span> | 
                  Requests/sec: <span className="text-purple-400">12,847</span>
                </div>
              </div>
            </div>
            
            {/* Floating badges */}
            <motion.div
              className="absolute -top-4 -right-4 glass px-4 py-2 rounded-xl text-sm font-medium flex items-center gap-2"
              animate={{ y: [0, -10, 0] }}
              transition={{ duration: 4, repeat: Infinity }}
            >
              <Box className="w-4 h-4 text-accent" />
              <span className="text-text-primary">Serverless Ready</span>
            </motion.div>
            <motion.div
              className="absolute -bottom-4 -left-4 glass px-4 py-2 rounded-xl text-sm font-medium flex items-center gap-2"
              animate={{ y: [0, 10, 0] }}
              transition={{ duration: 5, repeat: Infinity }}
            >
              <Layers className="w-4 h-4 text-purple-400" />
              <span className="text-text-primary">SOC2 Compliant</span>
            </motion.div>
          </motion.div>
        </motion.div>
      </div>

      {/* Scroll indicator */}
      <motion.div
        className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2"
        animate={{ y: [0, 10, 0] }}
        transition={{ duration: 2, repeat: Infinity }}
      >
        <span className="text-text-muted text-xs">Scroll to explore</span>
        <div className="w-6 h-10 rounded-full border-2 border-white/20 flex items-start justify-center p-1">
          <motion.div
            className="w-1.5 h-1.5 rounded-full bg-primary"
            animate={{ y: [0, 16, 0] }}
            transition={{ duration: 1.5, repeat: Infinity }}
          />
        </div>
      </motion.div>
    </section>
  )
}
