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

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Menu, X, Zap, ChevronRight, Terminal } from 'lucide-react'
import Link from 'next/link'
import { Button } from '@/components/ui/Button'
import { NAV_LINKS } from '@/lib/constants'

export default function Navbar() {
  const [isScrolled, setIsScrolled] = useState(false)
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false)

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 20)
    }

    window.addEventListener('scroll', handleScroll)
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  return (
    <>
      {/* Noise overlay */}
      <div className="noise-overlay" />
      
      <header
        className={`fixed top-0 left-0 right-0 z-50 transition-all duration-500 ${
          isScrolled ? 'glass-heavy py-4' : 'bg-transparent py-6'
        }`}
      >
        <div className="container-custom">
          <div className="flex items-center justify-between">
            {/* Logo */}
            <Link href="/" className="flex items-center gap-3 group">
              <div className="relative">
                <div className="w-12 h-12 bg-gradient-to-br from-primary via-purple-500 to-accent rounded-xl flex items-center justify-center group-hover:scale-110 transition-transform duration-300">
                  <Terminal className="w-6 h-6 text-white" />
                </div>
                <div className="absolute inset-0 bg-gradient-to-br from-primary via-purple-500 to-accent rounded-xl blur-lg opacity-50 group-hover:opacity-75 transition-opacity duration-300 animate-pulse" />
              </div>
              <div className="flex items-center gap-2">
                <span className="text-2xl font-bold gradient-text">KERNELIZE</span>
                <span className="px-2 py-0.5 text-xs font-medium bg-primary/20 text-primary-light rounded-full border border-primary/30">
                  v2.0
                </span>
              </div>
            </Link>

            {/* Desktop Navigation */}
            <nav className="hidden lg:flex items-center gap-1">
              {NAV_LINKS.map((link, index) => (
                <Link
                  key={link.href}
                  href={link.href}
                  className="group relative px-4 py-2 text-text-secondary hover:text-text-primary transition-all duration-200 font-medium"
                >
                  <span className="flex items-center gap-2">
                    {link.label}
                    {index === 0 && (
                      <span className="w-1.5 h-1.5 rounded-full bg-accent animate-pulse" />
                    )}
                  </span>
                  {/* Hover underline effect */}
                  <span className="absolute bottom-0 left-4 right-4 h-0.5 bg-gradient-to-r from-primary to-accent scale-x-0 group-hover:scale-x-100 transition-transform duration-300 origin-left" />
                </Link>
              ))}
            </nav>

            {/* Auth Buttons */}
            <div className="hidden lg:flex items-center gap-4">
              <Link href="/login">
                <Button variant="ghost" size="sm" className="hover:text-white">
                  Log In
                </Button>
              </Link>
              <Link href="/signup">
                <Button size="sm" className="group">
                  <span>Start Building</span>
                  <ChevronRight className="w-4 h-4 ml-1 group-hover:translate-x-1 transition-transform" />
                </Button>
              </Link>
            </div>

            {/* Mobile Menu Button */}
            <button
              className="lg:hidden p-3 text-text-secondary hover:text-text-primary transition-colors relative"
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
            >
              <div className="relative w-6 h-6">
                <motion.span
                  animate={{ rotate: isMobileMenuOpen ? 45 : 0, y: isMobileMenuOpen ? 0 : -8 }}
                  className="absolute left-0 w-full h-0.5 bg-current block"
                />
                <motion.span
                  animate={{ opacity: isMobileMenuOpen ? 0 : 1 }}
                  className="absolute left-0 w-full h-0.5 bg-current block top-1/2 -translate-y-1/2"
                />
                <motion.span
                  animate={{ rotate: isMobileMenuOpen ? -45 : 0, y: isMobileMenuOpen ? 0 : 8 }}
                  className="absolute left-0 w-full h-0.5 bg-current block"
                />
              </div>
            </button>
          </div>

          {/* Mobile Menu */}
          <AnimatePresence>
            {isMobileMenuOpen && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="lg:hidden mt-4 glass rounded-2xl overflow-hidden"
              >
                <nav className="flex flex-col p-4 gap-1">
                  {NAV_LINKS.map((link) => (
                    <Link
                      key={link.href}
                      href={link.href}
                      className="flex items-center justify-between px-4 py-3 text-text-secondary hover:text-text-primary hover:bg-white/5 rounded-xl transition-all duration-200 font-medium"
                      onClick={() => setIsMobileMenuOpen(false)}
                    >
                      {link.label}
                      <ChevronRight className="w-4 h-4" />
                    </Link>
                  ))}
                  <div className="flex flex-col gap-2 mt-4 pt-4 border-t border-white/10">
                    <Link href="/login" onClick={() => setIsMobileMenuOpen(false)}>
                      <Button variant="secondary" className="w-full">
                        Log In
                      </Button>
                    </Link>
                    <Link href="/signup" onClick={() => setIsMobileMenuOpen(false)}>
                      <Button className="w-full">
                        <Zap className="w-4 h-4 mr-2" />
                        Start Building Free
                      </Button>
                    </Link>
                  </div>
                </nav>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </header>

      {/* Scroll progress indicator */}
      <motion.div
        className="fixed top-0 left-0 right-0 h-1 bg-gradient-to-r from-primary via-purple-500 to-accent z-[60]"
        style={{ scaleX: isScrolled ? 1 : 0 }}
        transition={{ duration: 0.3 }}
      />
    </>
  )
}
