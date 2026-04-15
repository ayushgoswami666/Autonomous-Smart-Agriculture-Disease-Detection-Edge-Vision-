import { useState, useEffect } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { motion, useAnimation } from 'framer-motion'
import { useInView } from 'react-intersection-observer'
import {
  Leaf,
  ChevronDown,
  ArrowRight,
  CheckCircle2,
  AlertCircle,
  Users,
  BarChart3,
  Globe2,
  Mail,
  Phone,
  Linkedin,
  Twitter,
  Instagram,
  Menu,
  X,
  Globe
} from 'lucide-react'
import './LandingPage.css'
import heroImage from '../assets/hero_premium.png'

const StatsSection = () => {
  const stats = [
    { icon: <Leaf className="stat-icon" />, label: 'Farmers Empowered', value: 10000, suffix: '+' },
    { icon: <Users className="stat-icon" />, label: 'Agri Specialists', value: 500, suffix: '+' },
    { icon: <BarChart3 className="stat-icon" />, label: 'Productivity Improved', value: 35, suffix: '%' },
    { icon: <Globe2 className="stat-icon" />, label: 'Regions Impacted', value: 50, suffix: '+' }
  ]

  return (
    <section className="stats-section">
      <div className="container">
        <div className="stats-grid">
          {stats.map((stat, index) => (
            <Counter key={index} stat={stat} />
          ))}
        </div>
      </div>
    </section>
  )
}

const Counter = ({ stat }) => {
  const [count, setCount] = useState(0)
  const { ref, inView } = useInView({ triggerOnce: true, threshold: 0.5 })

  useEffect(() => {
    if (inView) {
      let start = 0
      const end = stat.value
      const duration = 2000
      const stepTime = 50
      const steps = duration / stepTime
      const increment = end / steps

      const timer = setInterval(() => {
        start += increment
        if (start >= end) {
          setCount(end)
          clearInterval(timer)
        } else {
          setCount(Math.floor(start))
        }
      }, stepTime)
      return () => clearInterval(timer)
    }
  }, [inView, stat.value])

  return (
    <div ref={ref} className="stat-card">
      <div className="stat-icon-wrapper">{stat.icon}</div>
      <div className="stat-value">{count}{stat.suffix}</div>
      <div className="stat-label">{stat.label}</div>
    </div>
  )
}

export default function LandingPage() {
  const [role, setRole] = useState('farmer')
  const [isMenuOpen, setIsMenuOpen] = useState(false)
  const [isScrolled, setIsScrolled] = useState(false)
  const navigate = useNavigate()

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 50)
    }
    window.addEventListener('scroll', handleScroll)
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  const handleRoleChange = (e) => {
    setRole(e.target.value)
  }

  const navigateToAuth = (mode) => {
    navigate(`/${role}/${mode}`)
  }

  return (
    <div className="landing-page">
      {/* Header */}
      <nav className={`navbar ${isScrolled ? 'navbar-scrolled' : ''}`}>
        <div className="container nav-container">

          {/* Left Section: Logo and Brand Name */}
          <Link to="/" className="brand">
            <Leaf className="logo-icon" />
            <span className="brand-name">Agri Vision</span>
          </Link>

          {/* Right Section: Navigation Controls */}
          <div className="nav-right">

            {/* Mobile Menu Toggle */}
            <button
              className="mobile-menu-toggle"
              onClick={() => setIsMenuOpen(!isMenuOpen)}
            >
              {isMenuOpen ? <X /> : <Menu />}
            </button>

            {/* Navigation Links */}
            <div className={`nav-links ${isMenuOpen ? 'nav-links-open' : ''}`}>
              <div className="role-selector-wrapper">
                <select
                  value={role}
                  onChange={handleRoleChange}
                  className="role-dropdown"
                >
                  <option value="farmer">Farmer</option>
                  <option value="specialist">AgriSpecialist</option>
                </select>
                <ChevronDown className="dropdown-arrow" />
              </div>

              <button
                onClick={() => navigateToAuth('signin')}
                className="btn-text"
              >
                Sign In
              </button>

              <button
                onClick={() => navigateToAuth('signup')}
                className="btn-primary-nav"
              >
                Sign Up
              </button>
            </div>
          </div>

        </div>
      </nav>

      {/* Hero Section */}
      <section className="hero">
        <div className="hero-bg">
          <img src={heroImage} alt="Modern Farming" />
          <div className="hero-overlay"></div>
        </div>
        <div className="container hero-content">
          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            Transforming Agriculture with <span className="highlight">Smart Digital Solutions</span>
          </motion.h1>
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
          >
            Agri Vision connects farmers with agricultural experts to enhance productivity, sustainability, and profitability.
          </motion.p>
          <motion.div
            className="hero-btns"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
          >
            <button onClick={() => navigateToAuth('signup')} className="btn-cta">Get Started <ArrowRight size={20} /></button>
            <a href="#about" className="btn-secondary">Learn More</a>
          </motion.div>
        </div>
      </section>

      {/* About Section */}
      <section id="about" className="about-section">
        <div className="container">
          <div className="section-header">
            <h2 className="section-title">About Agri Vision</h2>
            <div className="title-underline"></div>
          </div>
          <div className="about-content">
            <p>
              Agri Vision is an intelligent digital platform designed to empower farmers by connecting them with agricultural specialists.
              It provides expert guidance, data-driven insights, and innovative tools to improve crop yield and promote sustainable farming practices.
            </p>
          </div>
        </div>
      </section>

      {/* Problem & Solution */}
      <section className="problem-solution">
        <div className="container">
          <div className="grid-cols-2">
            <div className="card-box problems">
              <h3>The Challenges</h3>
              <ul>
                <li><AlertCircle className="icon-alert" /> Lack of expert agricultural guidance</li>
                <li><AlertCircle className="icon-alert" /> Unpredictable crop diseases</li>
                <li><AlertCircle className="icon-alert" /> Limited access to modern farming techniques</li>
                <li><AlertCircle className="icon-alert" /> Market uncertainty</li>
              </ul>
            </div>
            <div className="card-box solutions">
              <h3>Our Solutions</h3>
              <ul>
                <li><CheckCircle2 className="icon-check" /> Direct access to certified Agri Specialists</li>
                <li><CheckCircle2 className="icon-check" /> AI-driven insights and recommendations</li>
                <li><CheckCircle2 className="icon-check" /> Smart farming techniques and tools</li>
                <li><CheckCircle2 className="icon-check" /> Data-backed decision-making</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <StatsSection />

      {/* Benefits Section */}
      <section className="benefits">
        <div className="container">
          <div className="section-header">
            <h2 className="section-title">Benefits for You</h2>
            <div className="title-underline"></div>
          </div>
          <div className="grid-cols-2">
            <div className="benefit-card">
              <div className="benefit-header">
                <Users className="benefit-icon" />
                <h3>For Farmers</h3>
              </div>
              <ul className="benefit-list">
                <li>Expert guidance on crop management</li>
                <li>Increased productivity and profits</li>
                <li>Real-time solutions to agricultural challenges</li>
                <li>Access to modern farming techniques</li>
              </ul>
            </div>
            <div className="benefit-card">
              <div className="benefit-header">
                <Leaf className="benefit-icon" />
                <h3>For Agri Specialists</h3>
              </div>
              <ul className="benefit-list">
                <li>Opportunity to assist farmers globally</li>
                <li>Professional growth and recognition</li>
                <li>Data-driven insights and tools</li>
                <li>Seamless communication with farmers</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="final-cta">
        <div className="container cta-container">
          <motion.div
            className="cta-card"
            whileHover={{ scale: 1.02 }}
            transition={{ type: "spring", stiffness: 300 }}
          >
            <h2>Join the Future of Smart Agriculture</h2>
            <p>Empower your journey with Agri Vision today.</p>
            <div className="cta-buttons">
              <button
                onClick={() => { setRole('farmer'); navigateToAuth('signup'); }}
                className="btn-white"
              >
                Register as a Farmer
              </button>
              <button
                onClick={() => { setRole('specialist'); navigateToAuth('signup'); }}
                className="btn-outline-white"
              >
                Join as an Agri Specialist
              </button>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer className="footer">
        <div className="container">
          <div className="footer-grid">
            <div className="footer-info">
              <div className="footer-brand">
                <Leaf /> <span>Agri Vision</span>
              </div>
              <p>Empowering Agriculture Through Technology. We bridge the gap between tradition and innovation.</p>
              <div className="social-links">
                <a href="#"><Linkedin size={20} /></a>
                <a href="#"><Twitter size={20} /></a>
                <a href="#"><Globe size={20} /></a>
                <a href="#"><Instagram size={20} /></a>
              </div>
            </div>

            <div className="footer-links">
              <h4>Quick Links</h4>
              <ul>
                <li><a href="#">Home</a></li>
                <li><a href="#about">About</a></li>
                <li><a href="#">Features</a></li>
                <li><a href="#">Contact</a></li>
              </ul>
            </div>

            <div className="footer-links">
              <h4>Resources</h4>
              <ul>
                <li><a href="#">Blog</a></li>
                <li><a href="#">Help Center</a></li>
                <li><a href="#">Privacy Policy</a></li>
                <li><a href="#">Terms & Conditions</a></li>
              </ul>
            </div>

            <div className="footer-contact">
              <h4>Contact Us</h4>
              <ul>
                <li><Mail size={16} /> support@agrivision.com</li>
                <li><Phone size={16} /> +91-XXXXXXXXXX</li>
              </ul>
            </div>
          </div>

          <div className="footer-bottom">
            <p>&copy; 2025 Agri Vision. All Rights Reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  )
}
