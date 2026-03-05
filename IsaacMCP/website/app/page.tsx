import Navbar from "../components/Navbar";
import Hero from "../components/Hero";
import Features from "../components/Features";
import PluginPacks from "../components/PluginPacks";
import QuickStart from "../components/QuickStart";
import HowItWorks from "../components/HowItWorks";
import Demo from "../components/Demo";
import Install from "../components/Install";
import Security from "../components/Security";
import Deploy from "../components/Deploy";
import FAQ from "../components/FAQ";
import Footer from "../components/Footer";

export default function Home() {
  return (
    <>
      <Navbar />
      <Hero />
      <Features />
      <PluginPacks />
      <QuickStart />
      <HowItWorks />
      <Demo />
      <Install />
      <Security />
      <Deploy />
      <FAQ />
      <Footer />
    </>
  );
}
